#include "ptxsim/warp_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/ptx_config.h"
#include <algorithm>
#include <cassert>
#include <cstring>

WarpContext::WarpContext()
    : active_count(0), pc(0), warp_id(-1), single_step_mode(false),
      divergence_detected(false), sm_context_(nullptr) {
    // 初始化warp线程ID和活跃掩码
    for (int i = 0; i < WARP_SIZE; i++) {
        warp_thread_ids[i] = -1;
        active_mask[i] = false;
        pc_stacks[i] = std::vector<int>(); // 初始化PC栈
    }

    // 默认激活所有线程
    for (int i = 0; i < WARP_SIZE; i++) {
        active_mask[i] = true;
        warp_thread_ids[i] = i;
        pc_stacks[i].push_back(0); // 初始PC
    }
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
    // 根据活跃掩码执行指令
    for (int i = 0; i < WARP_SIZE; i++) {
        if (is_lane_active(i) && i < threads.size() && threads[i] != nullptr) {
            ThreadContext *thread = threads[i].get();

            // 设置线程的PC为当前warp的PC或线程自己的PC（用于处理分歧）
            if (!pc_stacks[i].empty()) {
                thread->set_pc(pc_stacks[i].back());
            }

            // 检查线程状态，如果是BAR_SYNC状态，说明线程在等待barrier，跳过执行
            if (thread->get_state() == BAR_SYNC) {
                continue; // 跳过处于barrier同步状态的线程
            }

            // 检查当前lane是否启用trace，以及trace_instruction_status是否启用
            if (ptxsim::DebugConfig::get().is_lane_traced(i) && 
                ptxsim::DebugConfig::get().is_trace_instruction_status_enabled()) {
                thread->print_instruction_status(stmt);
            }

            // 执行指令
            thread->execute_thread_instruction();

            // 检查线程状态，如果是BAR_SYNC状态，说明遇到了barrier指令
            if (thread->get_state() == BAR_SYNC && sm_context_ != nullptr) {
                // 在这里处理barrier同步
                // 遍历所有属于相同block的线程，执行同步
                sm_context_->synchronize_barrier(thread->bar_id,
                                                 thread); // 默认使用barrier 0
            }

            // 更新PC栈
            if (!pc_stacks[i].empty()) {
                pc_stacks[i].back() = thread->get_pc();
            } else {
                pc_stacks[i].push_back(thread->get_pc());
            }

            // 更新warp的PC为第一个活跃线程的PC（在SIMT模型中，所有活跃线程应该执行相同指令）
            if (i == 0 || pc == 0) {
                pc = thread->get_pc();
            }
        }
    }

    // 更新活跃掩码（例如，遇到分支指令时）
    update_active_mask();
}

void WarpContext::update_active_mask() {
    active_count = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
        if (i < threads.size() && threads[i] != nullptr) {
            // 检查线程状态，如果线程已退出则不再活跃
            if (threads[i]->is_exited()) {
                active_mask[i] = false;
            } else if (threads[i]->is_active()) {
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
    // 修改逻辑：不只是检查active_count，而是检查是否所有线程都已退出
    return is_all_threads_exited(); 
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