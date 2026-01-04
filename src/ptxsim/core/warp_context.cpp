#include "ptxsim/warp_context.h"
#include <algorithm>
#include <cassert>
#include <cstring>

WarpContext::WarpContext() 
    : active_count(0), pc(0), single_step_mode(false) {
    // 初始化warp线程ID和活跃掩码
    for (int i = 0; i < WARP_SIZE; i++) {
        warp_thread_ids[i] = -1;
        active_mask[i] = false;
    }
    
    // 默认激活所有线程
    for (int i = 0; i < WARP_SIZE; i++) {
        active_mask[i] = true;
        warp_thread_ids[i] = i;
    }
    active_count = WARP_SIZE;
}

void WarpContext::add_thread(ThreadContext* thread, int lane_id) {
    if (lane_id >= 0 && lane_id < WARP_SIZE) {
        threads.resize(std::max(threads.size(), static_cast<size_t>(lane_id + 1)));
        threads[lane_id] = thread;
        warp_thread_ids[lane_id] = thread ? thread->ThreadIdx.x + 
                                   thread->ThreadIdx.y * thread->BlockDim.x + 
                                   thread->ThreadIdx.z * thread->BlockDim.x * thread->BlockDim.y : -1;
    }
}

void WarpContext::execute_warp_instruction(StatementContext &stmt) {
    // 根据活跃掩码执行指令
    for (int i = 0; i < WARP_SIZE; i++) {
        if (active_mask[i] && i < threads.size() && threads[i] != nullptr) {
            ThreadContext* thread = threads[i];
            // 执行指令
            thread->handle_statement(stmt);
        }
    }
    
    // 更新活跃掩码（例如，遇到分支指令时）
    update_active_mask();
}

void WarpContext::update_active_mask() {
    active_count = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
        if (active_mask[i]) {
            active_count++;
        }
    }
}

bool WarpContext::is_complete() const {
    return active_count == 0;
}

void WarpContext::sync_warp() {
    // 在真实的硬件中，warp是同步执行的
    // 这里我们简单地同步所有线程的状态
    for (int i = 0; i < WARP_SIZE; i++) {
        if (active_mask[i] && i < threads.size() && threads[i] != nullptr) {
            // 确保所有活跃线程执行到相同PC
            threads[i]->pc = pc;
        }
    }
}

int WarpContext::get_active_count() const {
    return active_count;
}

bool WarpContext::has_active_threads() const {
    return active_count > 0;
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

void WarpContext::set_lane_active(int lane_id, bool active) {
    if (lane_id >= 0 && lane_id < WARP_SIZE) {
        bool was_active = active_mask[lane_id];
        active_mask[lane_id] = active;
        
        if (active && !was_active) {
            active_count++;
        } else if (!active && was_active) {
            active_count--;
        }
    }
}

bool WarpContext::is_lane_active(int lane_id) const {
    if (lane_id >= 0 && lane_id < WARP_SIZE) {
        return active_mask[lane_id];
    }
    return false;
}