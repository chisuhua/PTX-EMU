#include "ptxsim/warp_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/ptx_config.h"
#include <algorithm>
#include <cassert>
#include <cstring>

void WarpContext::handle_branch(const std::string& predicate,
                                 bool predicate_negated,
                                 int target_pc,
                                 int reconvergence_pc) {
    uint32_t taken_mask = 0;
    uint32_t not_taken_mask = 0;
    
    for (int i = 0; i < 32; i++) {
        bool should_branch = true;
        
        if (!predicate.empty()) {
            // Read predicate value from thread's register file
            std::string pred_name = predicate;
            if (!pred_name.empty() && pred_name[0] == '%') {
                pred_name = pred_name.substr(1);
            }
            
            // Get predicate register value from register bank
            if (register_bank_manager_) {
                void *reg_addr = register_bank_manager_->get_register(pred_name, warp_id, i);
                if (reg_addr) {
                    // Predicate registers store uint8_t values (0 or 1)
                    uint8_t pred_value = *static_cast<uint8_t*>(reg_addr);
                    bool pred_bool = (pred_value != 0);
                    should_branch = predicate_negated ? !pred_bool : pred_bool;
                }
            }
            // If register bank not available or register not found, assume true
        }
        
        if (should_branch) {
            taken_mask |= (1u << i);
        } else {
            not_taken_mask |= (1u << i);
        }
    }
    
    bool is_divergent = (taken_mask != 0) && (not_taken_mask != 0);
    
    if (is_divergent) {
        ptxsim::SIMTStackEntry entry;
        entry.branch_pc = pc;
        entry.reconvergence_pc = reconvergence_pc;
        entry.active_mask = taken_mask;
        entry.return_mask = warp_state.exec_mask;
        entry.return_pc = reconvergence_pc;
        
        simt_stack.push(entry);
        
        for (int i = 0; i < 32; i++) {
            if (taken_mask & (1u << i)) {
                warp_state.threads[i].pc = target_pc;
                warp_state.threads[i].next_pc = target_pc;
            } else if (not_taken_mask & (1u << i)) {
                warp_state.threads[i].pc = pc + 1;
                warp_state.threads[i].next_pc = pc + 1;
            }
        }
        
        warp_state.exec_mask = taken_mask;
    } else {
        int next_pc = (taken_mask != 0) ? target_pc : pc + 1;
        
        for (int i = 0; i < 32; i++) {
            if (warp_state.threads[i].is_active) {
                warp_state.threads[i].pc = next_pc;
                warp_state.threads[i].next_pc = next_pc;
            }
        }
    }
}

WarpContext::WarpContext()
    : active_count(0), pc(0), warp_id(-1), single_step_mode(false),
      divergence_detected(false), sm_context_(nullptr) {
    // 初始化 warp 线程 ID 和活跃掩码
    for (int i = 0; i < WARP_SIZE; i++) {
        warp_thread_ids[i] = -1;
        active_mask[i] = false;
        pc_stacks[i] = std::vector<int>(); // 初始化 PC 栈
    }

    // 默认激活所有线程
    for (int i = 0; i < WARP_SIZE; i++) {
        active_mask[i] = true;
        warp_thread_ids[i] = i;
        pc_stacks[i].push_back(0); // 初始 PC
        
        // 【SIMT Upgrade】初始化每线程状态
        warp_state.threads[i].pc = 0;
        warp_state.threads[i].next_pc = 0;
        warp_state.threads[i].is_active = true;
        warp_state.threads[i].is_exited = false;
        warp_state.threads[i].is_blocked = false;
        warp_state.threads[i].status = ptxsim::ThreadStatus::Active;
    }
    
    // 初始化执行掩码
    warp_state.exec_mask = 0xFFFFFFFF;
    
    active_count = WARP_SIZE;
}

void WarpContext::add_thread(std::unique_ptr<ThreadContext> thread,
                             int lane_id) {
    if (lane_id >= 0 && lane_id < WARP_SIZE) {
        threads.resize(
            std::max(threads.size(), static_cast<size_t>(lane_id + 1)));
        threads[lane_id] = std::move(thread);

        if (threads[lane_id]) {
            // 设置warp_context_指针
            threads[lane_id]->set_warp_context(this);

            warp_thread_ids[lane_id] =
                threads[lane_id]->ThreadIdx.x +
                threads[lane_id]->ThreadIdx.y * threads[lane_id]->BlockDim.x +
                threads[lane_id]->ThreadIdx.z * threads[lane_id]->BlockDim.x *
                    threads[lane_id]->BlockDim.y;
        } else {
            warp_thread_ids[lane_id] = -1;
        }
    }
}

void WarpContext::execute_warp_instruction(StatementContext &stmt) {
    if (stmt.type == S_BRA) {
        for (int i = 0; i < WARP_SIZE; i++) {
            if (is_lane_active(i) && i < threads.size() && threads[i] != nullptr) {
                ThreadContext *thread = threads[i].get();
                thread->sync_from_warp_state();
                thread->execute_thread_instruction();
                thread->sync_to_warp_state();
            }
        }
    } else {
        if (!simt_stack.empty()) {
            ;
        }
        
        for (int i = 0; i < WARP_SIZE; i++) {
            if (is_lane_active(i) && i < threads.size() && threads[i] != nullptr) {
                ThreadContext *thread = threads[i].get();
                thread->sync_from_warp_state();
                
                if (thread->get_state() == BAR_SYNC) {
                    if (sm_context_ != nullptr) {
                        bool is_warp_barrier = (warp_state.current_wbar_id >= 0);
                        bool warp_barrier_complete = is_warp_barrier &&
                            warp_state.wbars[warp_state.current_wbar_id].is_complete();

                        if (!warp_barrier_complete) {
                            sm_context_->synchronize_barrier(thread->bar_id, thread);
                        }
                    }
                    thread->sync_to_warp_state();
                    continue;
                }
                
                thread->execute_thread_instruction();
                thread->sync_to_warp_state();
            }
        }
    }
    
    update_active_mask();
}

void WarpContext::update_active_mask() {
    active_count = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
        if (i < threads.size() && threads[i] != nullptr) {
            if (threads[i]->is_exited()) {
                active_mask[i] = false;
            } else {
                active_mask[i] = true;
                active_count++;
            }
        }
    }
}

void WarpContext::set_active_mask(int lane_id, bool active) {
    if (lane_id >= 0 && lane_id < WARP_SIZE) {
        bool was_active = active_mask[lane_id];
        active_mask[lane_id] = active;

        if (was_active && !active) {
            active_count--;
        } else if (!was_active && active) {
            active_count++;
        }
    }
}

bool WarpContext::is_finished() const {
    // A warp is finished when active_count is 0 (no active threads)
    // Note: This is independent of is_all_threads_exited() which checks
    // the threads vector. A default WarpContext has active_count=32 but
    // an empty threads vector until threads are properly added.
    return active_count == 0;
}

bool WarpContext::is_all_threads_exited() const {
    // 检查warp中的所有线程是否都已退出
    for (int i = 0; i < WARP_SIZE; i++) {
        if (i < threads.size() && threads[i] != nullptr) {
            if (!threads[i]->is_exited()) {
                // 如果有任何一个线程还没有退出，则warp尚未完成
                return false;
            }
        }
    }
    return true;
}

void WarpContext::sync_threads() {
    // 在真正的硬件模拟中，这里会实现warp级同步
    // 目前我们简单地确保所有活跃线程都执行到相同的PC
}

void WarpContext::reset() {
    for (int i = 0; i < WARP_SIZE; i++) {
        active_mask[i] = true;
        if (i < threads.size() && threads[i] != nullptr) {
            threads[i]->reset();
        }
        // 重置PC栈
        pc_stacks[i].clear();
        pc_stacks[i].push_back(0);
    }
    active_count = WARP_SIZE;
    pc = 0;
    divergence_detected = false;
}

void WarpContext::handle_branch_divergence(int lane_id, int new_pc) {
    if (lane_id >= 0 && lane_id < WARP_SIZE) {
        // 将当前PC压入栈中
        if (!pc_stacks[lane_id].empty()) {
            pc_stacks[lane_id].push_back(pc_stacks[lane_id].back());
        } else {
            pc_stacks[lane_id].push_back(0);
        }

        // 设置新PC
        pc_stacks[lane_id].back() = new_pc;

        divergence_detected = true;
    }
}

uint32_t WarpContext::get_active_mask() const {
    uint32_t mask = 0;
    for (int i = 0; i < WARP_SIZE && i < 32; i++) {
        if (active_mask[i]) {
            mask |= (1U << i);
        }
    }
    return mask;
}

void WarpContext::set_active_mask(uint32_t mask) {
    active_count = 0;
    for (int i = 0; i < WARP_SIZE && i < 32; i++) {
        bool active = (mask >> i) & 1;
        active_mask[i] = active;
        if (active) {
            active_count++;
        }
    }
}