#ifndef WBAR_H
#define WBAR_H

#include <cstdint>

namespace ptxsim {

/**
 * @file wbar.h
 * @brief Warp 级收敛屏障 (Warp Barrier)
 * @details 用于解决 warp 级自旋锁死锁问题
 *          支持 PTX ISA v6.0+ 的 bar.warp.sync 指令
 * @author PTX-EMU Team
 * @date 2026-04-02
 * 
 * 使用场景:
 * 1. 分支发散前：.arrive 注册参与线程
 * 2. 分支汇合点：.wait 等待所有线程
 * 
 * 硬件对应:
 * - NVIDIA bar.warp.sync 指令 (PTX ISA v6.0+)
 * - 4 个屏障寄存器 (典型硬件配置)
 */

struct Wbar {
    // 参与掩码：在 .arrive 时设置，标记哪些线程参与此屏障
    uint32_t participation_mask = 0;
    
    // 到达掩码：线程到达 .wait 时设置
    uint32_t arrived_mask = 0;
    
    // 汇合点 PC：屏障解除后跳转的 PC
    int reconvergence_pc = -1;
    
    // 状态标志
    bool is_initialized = false;
    
    // 期望到达计数 (用于调试/验证)
    int expected_count = 0;
    
    // 重置屏障
    void reset() {
        participation_mask = 0;
        arrived_mask = 0;
        reconvergence_pc = -1;
        is_initialized = false;
        expected_count = 0;
    }
    
    // 检查是否所有参与线程都已到达
    bool is_complete() const {
        if (!is_initialized || participation_mask == 0) {
            return false;
        }
        return (arrived_mask & participation_mask) == participation_mask;
    }
    
    // 获取参与线程数量
    int count_participants() const {
        return __builtin_popcount(participation_mask);
    }
    
    // 获取已到达线程数量
    int count_arrived() const {
        return __builtin_popcount(arrived_mask);
    }
    
    // 标记线程到达
    void arrive(int lane_id) {
        if (lane_id >= 0 && lane_id < 32) {
            arrived_mask |= (1u << lane_id);
        }
    }
    
    // 设置参与掩码 (在 .arrive 时调用)
    void set_participants(uint32_t mask) {
        participation_mask = mask;
        expected_count = __builtin_popcount(mask);
    }
    
    // 设置汇合点 PC
    void set_reconvergence_pc(int pc) {
        reconvergence_pc = pc;
    }
    
    // 初始化屏障
    void init(uint32_t participants, int reconvergence_pc) {
        reset();
        participation_mask = participants;
        expected_count = __builtin_popcount(participants);
        this->reconvergence_pc = reconvergence_pc;
        is_initialized = true;
    }
};

} // namespace ptxsim

#endif // WBAR_H
