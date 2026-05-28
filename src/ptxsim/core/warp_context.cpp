#include "ptxsim/warp_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/ptx_config.h"
#include "ptxsim/execution_trace.h"
#include "ptxsim/warp_trace_formatter.h"
#include <algorithm>
#include <cassert>
#include <cstring>

void WarpContext::handle_branch(const std::string& predicate,
                                 bool predicate_negated,
                                 int target_pc,
                                 int reconvergence_pc,
                                 int current_inst_pc) {
    assert(current_inst_pc >= 0 && "handle_branch: current_inst_pc must be non-negative, got default -1");
    uint32_t taken_mask = 0;
    uint32_t not_taken_mask = 0;
    
    for (int i = 0; i < 32; i++) {
        if (!warp_state.threads[i].is_active) continue;
        if (warp_state.threads[i].pc != current_inst_pc) continue;

        bool should_branch = true;

        if (!predicate.empty()) {
            std::string pred_name = predicate;
            if (!pred_name.empty() && pred_name[0] == '%') {
                pred_name = pred_name.substr(1);
            }

            if (register_bank_manager_) {
                void *reg_addr = register_bank_manager_->get_register(pred_name, warp_id, i);
                if (reg_addr) {
                    uint8_t pred_value = *static_cast<uint8_t*>(reg_addr);
                    bool pred_bool = (pred_value != 0);
                    should_branch = predicate_negated ? !pred_bool : pred_bool;
                }
            }
        }

        if (should_branch) {
            taken_mask |= (1u << i);
        } else {
            not_taken_mask |= (1u << i);
        }
    }

    bool is_divergent = (taken_mask != 0) && (not_taken_mask != 0);
    int fallthrough_pc = current_inst_pc + 1;

    if (is_divergent) {
        ptxsim::SIMTStackEntry entry;
        entry.branch_pc = current_inst_pc;
        entry.reconvergence_pc = reconvergence_pc;
        entry.active_mask = taken_mask;
        entry.return_mask = warp_state.exec_mask;
        entry.return_pc = reconvergence_pc;

        simt_stack.push(entry);

        // SIMT栈push跟踪
        if (ptxsim::DebugConfig::get().is_trace_simt_stack_enabled() && sm_context_) {
            PTX_DEBUG_EMU("%s", ptxsim::WarpTraceFormatter::format_simt_push(
                sm_context_->get_cycle_count(), sm_context_->get_sm_id(), warp_id,
                entry, taken_mask, simt_stack).c_str());
        }

        for (int i = 0; i < 32; i++) {
            if (taken_mask & (1u << i)) {
                warp_state.threads[i].pc = target_pc;
                warp_state.threads[i].next_pc = target_pc;
            } else if (not_taken_mask & (1u << i)) {
                warp_state.threads[i].pc = fallthrough_pc;
                warp_state.threads[i].next_pc = fallthrough_pc;
            }
        }

        warp_state.exec_mask = taken_mask;
    } else {
        int next_pc = (taken_mask != 0) ? target_pc : fallthrough_pc;

        for (int i = 0; i < 32; i++) {
            if (warp_state.threads[i].is_active &&
                warp_state.threads[i].pc == current_inst_pc) {
                warp_state.threads[i].pc = next_pc;
                warp_state.threads[i].next_pc = next_pc;
            }
        }
    }
}

void WarpContext::advance_thread_pc(int lane_id, int new_pc) {
    if (lane_id < 0 || lane_id >= WARP_SIZE) return;
    warp_state.threads[lane_id].pc = new_pc;
    warp_state.threads[lane_id].next_pc = new_pc;
}

bool WarpContext::check_reconvergence() {
    if (simt_stack.empty()) return false;

    size_t depth_before = simt_stack.depth();
    
    // 检查是否收敛，如果收敛则记录被弹出的条目用于跟踪
    ptxsim::SIMTStackEntry popped_entry;
    bool will_pop = simt_stack.top().is_converged(warp_state.threads);
    if (will_pop) {
        popped_entry = simt_stack.top();
    }
    
    simt_stack.check_reconvergence(warp_state.threads);

    if (simt_stack.depth() < depth_before) {
        if (simt_stack.empty()) {
            warp_state.exec_mask = 0xFFFFFFFF;
        } else {
            warp_state.exec_mask = simt_stack.top().active_mask;
        }
        // SIMT栈pop跟踪
        if (ptxsim::DebugConfig::get().is_trace_simt_stack_enabled() && sm_context_) {
            PTX_DEBUG_EMU("%s", ptxsim::WarpTraceFormatter::format_simt_pop(
                sm_context_->get_cycle_count(), sm_context_->get_sm_id(), warp_id,
                popped_entry).c_str());
        }
        return true;
    }
    return false;
}

WarpContext::WarpContext()
    : active_count(0), pc(0), warp_id(-1), single_step_mode(false),
      divergence_detected(false), sm_context_(nullptr) {
    for (int i = 0; i < WARP_SIZE; i++) {
        warp_thread_ids[i] = -1;
        active_mask[i] = true;
        warp_state.threads[i].pc = 0;
        warp_state.threads[i].next_pc = 0;
        warp_state.threads[i].is_active = true;
        warp_state.threads[i].is_exited = false;
        warp_state.threads[i].is_blocked = false;
        warp_state.threads[i].status = ptxsim::ThreadStatus::Active;
    }
    warp_state.exec_mask = 0xFFFFFFFF;
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
            
            warp_state.threads[lane_id].is_active = true;
            active_mask[lane_id] = true;
            active_count++;
        } else {
            warp_thread_ids[lane_id] = -1;
        }
    }
}

void WarpContext::execute_warp_instruction(StatementContext &stmt, int target_pc) {
    for (int i = 0; i < WARP_SIZE; i++) {
        if (i >= threads.size() || threads[i] == nullptr) {
            continue;
        }

        ThreadContext *thread = threads[i].get();
        bool lane_active = is_lane_active(i);
        bool blocked_at_barrier = (thread->get_state() == BAR_SYNC);

        // Hybrid fix: allow blocked threads to enter even if not lane_active
        // so BAR_SYNC fallback handling can unblock them
        if (!lane_active && !blocked_at_barrier) {
            continue;
        }
        
        // Only execute for lanes at the target PC.
        // Must use warp_state.pc to match get_lanes_by_pc() source.
        if (warp_state.threads[i].pc != static_cast<uint32_t>(target_pc)) {
            continue;
        }
        
        thread->sync_from_warp_state();
        
        if (thread->get_state() == BAR_SYNC) {
            if (sm_context_ != nullptr) {
                bool is_warp_barrier = (warp_state.current_wbar_id >= 0);
                bool warp_barrier_complete = is_warp_barrier &&
                    warp_state.wbars[warp_state.current_wbar_id].is_complete();

                if (!warp_barrier_complete) {
                    PTX_WARN_EMU("Fallback CTA sync: lane %d, wbar %d incomplete",
                                  thread->lane_id_, warp_state.current_wbar_id);
                    sm_context_->synchronize_barrier(thread->bar_id, thread);
                }
            }
            thread->sync_to_warp_state();
            continue;
        }
        
        thread->execute_thread_instruction();
        thread->sync_to_warp_state();

        if (ptxsim::ExecutionTracer::is_enabled()) {
          ptxsim::ExecutionTracer::record(
              i, warp_state.threads[i].pc, stmt.instructionText);
        }
    }
    
    update_active_mask();
}

void WarpContext::update_active_mask() {
    active_count = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
        if (i < threads.size() && threads[i] != nullptr) {
            bool active = warp_state.threads[i].is_active &&
                          !warp_state.threads[i].is_exited &&
                          !warp_state.threads[i].is_blocked &&
                          (warp_state.threads[i].status == ptxsim::ThreadStatus::Active);
            active_mask[i] = active;
            warp_state.threads[i].is_active = active;
            if (active) active_count++;
        }
    }
}

void WarpContext::set_active_mask(int lane_id, bool active) {
    if (lane_id >= 0 && lane_id < WARP_SIZE) {
        bool was_active = active_mask[lane_id];
        active_mask[lane_id] = active;
        warp_state.threads[lane_id].is_active = active;

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

bool WarpContext::is_warp_ready_to_fetch() const {
    for (int i = 0; i < WARP_SIZE; i++) {
        if (!warp_state.threads[i].is_active) continue;
        if (warp_state.threads[i].pc != warp_state.threads[i].next_pc) {
            return false;
        }
    }
    return true;
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
    active_count = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
        if (i < threads.size() && threads[i] != nullptr) {
            threads[i]->reset();
            active_mask[i] = true;
            warp_state.threads[i].is_active = true;
            active_count++;
        } else {
            active_mask[i] = false;
            warp_state.threads[i].is_active = false;
        }
    }
    pc = 0;
    divergence_detected = false;
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
        warp_state.threads[i].is_active = active;
        if (active) {
            active_count++;
        }
    }
}

std::map<int, std::vector<int>> WarpContext::get_lanes_by_pc() const {
    std::map<int, std::vector<int>> pc_to_lanes;

    for (int lane = 0; lane < WARP_SIZE; lane++) {
        if (lane < (int)threads.size() && threads[lane] != nullptr &&
            warp_state.threads[lane].is_active && 
            !warp_state.threads[lane].is_exited) {
            int pc = warp_state.threads[lane].pc;
            pc_to_lanes[pc].push_back(lane);
        }
    }

    return pc_to_lanes;
}

std::vector<int> WarpContext::get_unique_pcs() const {
    std::vector<int> pcs;
    auto lanes_by_pc = get_lanes_by_pc();

    for (const auto& [pc, lanes] : lanes_by_pc) {
        pcs.push_back(pc);
    }

    return pcs;
}

void WarpContext::force_reconvergence_at_barrier(int barrier_pc) {
    // 不主动推进线程PC —— 让调度器自然选择非阻塞的PC执行
    // 屏障处理器会在 divergence 路径中阻塞当前线程（set is_blocked=true），
    // 调度器随后会跳过有阻塞线程的PC组，选择其他PC执行。
    // 当所有线程都到达屏障后，wbar 完成并释放所有线程。
    // 
    // 注意：不能推进线程PC，否则会跳过 shared memory store 等关键指令。
    // 注释掉的代码（advance_thread_pc）曾导致 E2E 测试中共享内存数据丢失。
}
