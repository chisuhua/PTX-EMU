#ifndef WARP_SCHEDULER_H
#define WARP_SCHEDULER_H

#include "warp_context.h"
#include <vector>
#include <queue>

class WarpScheduler {
public:
    virtual ~WarpScheduler() = default;
    
    // 添加warp到调度器
    virtual void add_warp(WarpContext* warp) = 0;
    
    // 调度下一个warp执行
    virtual WarpContext* schedule_next() = 0;
    
    // 更新调度器状态
    virtual void update_state() = 0;
    
    // 检查是否所有warp都已完成
    virtual bool all_warps_finished() const = 0;
};

// 轮询调度器实现
class RoundRobinWarpScheduler : public WarpScheduler {
public:
    RoundRobinWarpScheduler() = default;
    virtual ~RoundRobinWarpScheduler() = default;
    
    void add_warp(WarpContext* warp) override;
    WarpContext* schedule_next() override;
    void update_state() override;
    bool all_warps_finished() const override;
    
private:
    std::vector<WarpContext*> warps;
    size_t current_warp_idx = 0;
};

// 贪心调度器实现
class GreedyWarpScheduler : public WarpScheduler {
public:
    GreedyWarpScheduler() = default;
    virtual ~GreedyWarpScheduler() = default;
    
    void add_warp(WarpContext* warp) override;
    WarpContext* schedule_next() override;
    void update_state() override;
    bool all_warps_finished() const override;
    
private:
    std::vector<WarpContext*> warps;
    std::queue<WarpContext*> ready_warps;
};

#endif // WARP_SCHEDULER_H