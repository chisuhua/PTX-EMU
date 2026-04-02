#ifndef WARP_STATE_H
#define WARP_STATE_H

#include "ptxsim/thread_state.h"
#include "ptxsim/wbar.h"
#include <array>
#include <cstdint>

namespace ptxsim {

/**
 * @file warp_state.h
 * @brief Warp 级状态管理，支持每线程 PC 和收敛屏障
 * @details 这是 SIMT 架构升级的核心数据结构
 * @author PTX-EMU Team
 * @date 2026-04-02
 */

// Warp 级状态结构
struct WarpState {
    // 每线程状态数组 (32 lanes)
    std::array<ThreadState, 32> threads;
    
    // 执行掩码寄存器 (快速查询活跃 lanes)
    uint32_t exec_mask = 0xFFFFFFFF;  // 初始全活跃
    
    // Warp 级屏障寄存器 (4 个，硬件典型值)
    std::array<Wbar, 4> wbars;
    int current_wbar_id = -1;  // 当前使用的 wbar ID
    
    // Warp PC (仅用于调试/兼容，实际执行使用 per-thread PC)
    uint32_t warp_pc = 0;
    
    // 分支栈 (用于传统收敛，逐步淘汰)
    std::array<int, 16> pc_stack;
    int pc_stack_depth = 0;
    
    // 重置 warp 状态
    void reset() {
        for (auto& thread : threads) {
            thread.reset();
        }
        exec_mask = 0xFFFFFFFF;
        for (auto& wbar : wbars) {
            wbar.reset();
        }
        current_wbar_id = -1;
        warp_pc = 0;
        pc_stack_depth = 0;
    }
    
    // 获取活跃 lane 数量
    int count_active_lanes() const {
        int count = 0;
        for (int i = 0; i < 32; ++i) {
            if (threads[i].is_active && !threads[i].is_exited) {
                ++count;
            }
        }
        return count;
    }
    
    // 获取可调度的 lane 数量
    int count_schedulable_lanes() const {
        int count = 0;
        for (int i = 0; i < 32; ++i) {
            if (threads[i].is_schedulable()) {
                ++count;
            }
        }
        return count;
    }
    
    // 检查 warp 是否全部退出
    bool is_all_exited() const {
        for (const auto& thread : threads) {
            if (!thread.is_exited) {
                return false;
            }
        }
        return true;
    }
    
    // 检查 warp 是否有可调度的线程
    bool has_schedulable_threads() const {
        return count_schedulable_lanes() > 0;
    }
};

} // namespace ptxsim

#endif // WARP_STATE_H
