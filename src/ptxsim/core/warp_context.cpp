#include "ptxsim/warp_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/ptx_config.h"
#include <algorithm>
#include <cassert>
#include <cstring>

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
#ifdef PTX_DEBUG_WARP_VERBOSE
    // DEBUG: Log warp-level state at start of each instruction execution
    PTX_INFO_EMU("=== execute_warp_instruction: threads.size()=%zu active_count=%d ===",
                  threads.size(), active_count);
#endif
        for (int i = 0; i < WARP_SIZE; i++) {
        if (is_lane_active(i) && i < threads.size() && threads[i] != nullptr) {
            ThreadContext *thread = threads[i].get();
            
            // Get thread's PC from its own pc_stack
            if (!pc_stacks[i].empty()) {
                thread->set_pc(pc_stacks[i].back());
            }
            
            // Check thread state - if BAR_SYNC, thread is waiting at barrier
            if (thread->get_state() == BAR_SYNC) {
                if (sm_context_ != nullptr) {
                    sm_context_->synchronize_barrier(thread->bar_id, thread);
                }
                continue;
            }
            
#ifdef PTX_DEBUG_WARP_VERBOSE
            // DEBUG: Log lane execution details
            const char* state_str = "UNKNOWN";
            switch (thread->get_state()) {
                case IDLE: state_str = "IDLE"; break;
                case RUN: state_str = "RUN"; break;
                case EXIT: state_str = "EXIT"; break;
                case BAR_SYNC: state_str = "BAR_SYNC"; break;
            }
            PTX_INFO_EMU("lane=%d is_lane_active=%d thread=0x%llx state=%s pc=%d",
                         i, is_lane_active(i), (unsigned long long)(uintptr_t)thread, state_str, thread->get_pc());
#endif

            // Execute the instruction at thread's current PC
            thread->execute_thread_instruction();
            
            // Update PC stack with thread's new PC
            if (!pc_stacks[i].empty()) {
                pc_stacks[i].back() = thread->get_pc();
            } else {
                pc_stacks[i].push_back(thread->get_pc());
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