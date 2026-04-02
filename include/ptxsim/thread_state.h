#ifndef THREAD_STATE_H
#define THREAD_STATE_H

#include <cstdint>
#include <array>

namespace ptxsim {

/**
 * @file thread_state.h
 * @brief Per-thread state for SIMT execution model
 * @details 支持每线程 PC 和独立的线程状态管理，用于解决 warp 级自旋锁死锁问题
 * @author PTX-EMU Team
 * @date 2026-04-02
 * 
 * 这是 SIMT 架构升级的核心数据结构，实现了：
 * 1. 每线程 PC - 允许 divergent 线程独立推进
 * 2. 活跃状态 - 跟踪线程是否可执行
 * 3. 阻塞状态 - 用于 barrier 等待
 * 4. 退出状态 - 标记线程是否已退出
 */

// 线程状态枚举
enum class ThreadStatus : uint8_t {
    Active,     // 线程可执行
    Blocked,    // 线程在 barrier 等待
    Exited,     // 线程已退出
    Yielded     // 线程主动让出 (用于长延迟操作)
};

// 每线程状态结构
struct ThreadState {
    // PC 相关
    uint32_t pc = 0;          // 当前程序计数器
    uint32_t next_pc = 0;     // 下一条指令 PC
    
    // 状态标志
    ThreadStatus status = ThreadStatus::Active;
    bool is_exited = false;   // 是否已退出 (permanent)
    
    // Barrier 相关
    bool is_blocked = false;  // 是否在 barrier 等待
    
    // 执行掩码 (用于快速查询)
    bool is_active = true;    // 是否活跃 (可调度)
    
    // 重置线程状态
    void reset() {
        pc = 0;
        next_pc = 0;
        status = ThreadStatus::Active;
        is_exited = false;
        is_blocked = false;
        is_active = true;
    }
    
    // 检查线程是否可调度的快捷方法
    bool is_schedulable() const {
        return is_active && !is_exited && !is_blocked && (status == ThreadStatus::Active);
    }
};

// 辅助函数：将 ThreadStatus 转为字符串
inline const char* thread_status_to_string(ThreadStatus status) {
    switch (status) {
        case ThreadStatus::Active:   return "Active";
        case ThreadStatus::Blocked:  return "Blocked";
        case ThreadStatus::Exited:   return "Exited";
        case ThreadStatus::Yielded:  return "Yielded";
        default:                     return "Unknown";
    }
}

} // namespace ptxsim

#endif // THREAD_STATE_H
