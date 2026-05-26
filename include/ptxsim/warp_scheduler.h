#ifndef WARP_SCHEDULER_H
#define WARP_SCHEDULER_H

#include "warp_context.h"
#include "ptxsim/bsync_state.h"
#include <vector>
#include <queue>

class WarpScheduler {
public:
    virtual ~WarpScheduler() = default;

    virtual void add_warp(WarpContext* warp) = 0;
    virtual void remove_warp(WarpContext* warp) = 0;
    virtual WarpContext* schedule_next() = 0;
    virtual void update_state() = 0;
    virtual bool all_warps_finished() const = 0;

    virtual void set_execution_mode(ptxsim::DivergenceExecutionMode mode) = 0;
    virtual ptxsim::DivergenceExecutionMode get_execution_mode() const = 0;
    virtual bool schedule_with_migration(WarpContext* warp) = 0;
};

class RoundRobinWarpScheduler : public WarpScheduler {
public:
    RoundRobinWarpScheduler() = default;
    virtual ~RoundRobinWarpScheduler() = default;

    void add_warp(WarpContext* warp) override;
    void remove_warp(WarpContext* warp) override;
    WarpContext* schedule_next() override;
    void update_state() override;
    bool all_warps_finished() const override;
    void set_execution_mode(ptxsim::DivergenceExecutionMode mode) override;
    ptxsim::DivergenceExecutionMode get_execution_mode() const override;
    bool schedule_with_migration(WarpContext* warp) override;

private:
    std::vector<WarpContext*> warps;
    size_t current_warp_idx = 0;
    ptxsim::DivergenceExecutionMode execution_mode_ = ptxsim::DivergenceExecutionMode::Sequential;
};

class GreedyWarpScheduler : public WarpScheduler {
public:
    GreedyWarpScheduler() = default;
    virtual ~GreedyWarpScheduler() = default;

    void add_warp(WarpContext* warp) override;
    void remove_warp(WarpContext* warp) override;
    WarpContext* schedule_next() override;
    void update_state() override;
    bool all_warps_finished() const override;
    void set_execution_mode(ptxsim::DivergenceExecutionMode mode) override;
    ptxsim::DivergenceExecutionMode get_execution_mode() const override;
    bool schedule_with_migration(WarpContext* warp) override;

private:
    std::vector<WarpContext*> warps;
    std::queue<WarpContext*> ready_warps;
    ptxsim::DivergenceExecutionMode execution_mode_ = ptxsim::DivergenceExecutionMode::Sequential;
};

#endif // WARP_SCHEDULER_H