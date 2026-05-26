#include "ptxsim/warp_scheduler.h"
#include "ptxsim/bsync_state.h"
#include <algorithm>

using namespace ptxsim;

void RoundRobinWarpScheduler::add_warp(WarpContext* warp) {
    warps.push_back(warp);
}

void RoundRobinWarpScheduler::remove_warp(WarpContext* warp) {
    auto it = std::find(warps.begin(), warps.end(), warp);
    if (it != warps.end()) {
        warps.erase(it);
    }
}

WarpContext* RoundRobinWarpScheduler::schedule_next() {
    if (warps.empty()) {
        return nullptr;
    }

    // 寻找下一个活跃的warp
    size_t start_idx = current_warp_idx;
    do {
        WarpContext* warp = warps[current_warp_idx];
        if (warp && warp->is_active() && !warp->is_finished() &&
            warp->is_warp_ready_to_fetch()) {
            current_warp_idx = (current_warp_idx + 1) % warps.size();
            return warp;
        }
        current_warp_idx = (current_warp_idx + 1) % warps.size();
    } while (current_warp_idx != start_idx);

    return nullptr;  // 没有就绪的warp
}

void RoundRobinWarpScheduler::update_state() {
    // 更新状态，例如清理已完成的warp
}

bool RoundRobinWarpScheduler::all_warps_finished() const {
    for (auto* warp : warps) {
        if (warp && !warp->is_finished()) {
            return false;
        }
    }
    return true;
}

void GreedyWarpScheduler::add_warp(WarpContext* warp) {
    warps.push_back(warp);
}

void GreedyWarpScheduler::remove_warp(WarpContext* warp) {
    auto it = std::find(warps.begin(), warps.end(), warp);
    if (it != warps.end()) {
        warps.erase(it);
    }
}

WarpContext* GreedyWarpScheduler::schedule_next() {
    // 首先尝试从ready队列中获取warp
    if (!ready_warps.empty()) {
        WarpContext* warp = ready_warps.front();
        ready_warps.pop();
        if (warp && warp->is_active() && !warp->is_finished() &&
            warp->is_warp_ready_to_fetch()) {
            return warp;
        }
        // 如果warp不再就绪，继续寻找
    }

    // 遍历所有warp找到下一个就绪的
    for (auto* warp : warps) {
        if (warp && warp->is_active() && !warp->is_finished() &&
            warp->is_warp_ready_to_fetch()) {
            return warp;
        }
    }

    return nullptr;
}

void GreedyWarpScheduler::update_state() {
    // 将所有活跃的warp加入ready队列
    ready_warps = std::queue<WarpContext*>();
    for (auto* warp : warps) {
        if (warp && warp->is_active() && !warp->is_finished()) {
            ready_warps.push(warp);
        }
    }
}

bool GreedyWarpScheduler::all_warps_finished() const {
    for (auto* warp : warps) {
        if (warp && !warp->is_finished()) {
            return false;
        }
    }
    return true;
}

void RoundRobinWarpScheduler::set_execution_mode(DivergenceExecutionMode mode) {
    execution_mode_ = mode;
}

DivergenceExecutionMode RoundRobinWarpScheduler::get_execution_mode() const {
    return execution_mode_;
}

bool RoundRobinWarpScheduler::schedule_with_migration(WarpContext* warp) {
    if (!warp || warps.empty()) {
        return false;
    }

    if (execution_mode_ == DivergenceExecutionMode::Sequential) {
        return false;
    }

    size_t start_idx = current_warp_idx;
    do {
        WarpContext* w = warps[current_warp_idx];
        if (w && w != warp && w->is_active() && !w->is_finished() &&
            w->is_warp_ready_to_fetch()) {
            if (execution_mode_ == DivergenceExecutionMode::Interleaved) {
                if (rand() % 2 == 0) {
                    current_warp_idx = (current_warp_idx + 1) % warps.size();
                    return true;
                }
            } else if (execution_mode_ == DivergenceExecutionMode::ShortestFirst) {
                current_warp_idx = (current_warp_idx + 1) % warps.size();
                return true;
            }
        }
        current_warp_idx = (current_warp_idx + 1) % warps.size();
    } while (current_warp_idx != start_idx);

    return false;
}

void GreedyWarpScheduler::set_execution_mode(DivergenceExecutionMode mode) {
    execution_mode_ = mode;
}

DivergenceExecutionMode GreedyWarpScheduler::get_execution_mode() const {
    return execution_mode_;
}

bool GreedyWarpScheduler::schedule_with_migration(WarpContext* warp) {
    if (!warp || warps.empty()) {
        return false;
    }

    if (execution_mode_ == DivergenceExecutionMode::Sequential) {
        return false;
    }

    for (auto* w : warps) {
        if (w && w != warp && w->is_active() && !w->is_finished() &&
            w->is_warp_ready_to_fetch()) {
            if (execution_mode_ == DivergenceExecutionMode::Interleaved) {
                if (rand() % 2 == 0) {
                    return true;
                }
            } else if (execution_mode_ == DivergenceExecutionMode::ShortestFirst) {
                return true;
            }
        }
    }

    return false;
}