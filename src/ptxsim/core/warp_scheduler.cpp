#include "ptxsim/warp_scheduler.h"
#include <algorithm>

void RoundRobinWarpScheduler::add_warp(WarpContext* warp) {
    warps.push_back(warp);
}

WarpContext* RoundRobinWarpScheduler::schedule_next() {
    if (warps.empty()) {
        return nullptr;
    }
    
    // 寻找下一个活跃的warp
    size_t start_idx = current_warp_idx;
    do {
        WarpContext* warp = warps[current_warp_idx];
        if (warp && warp->is_active() && !warp->is_finished()) {
            current_warp_idx = (current_warp_idx + 1) % warps.size();
            return warp;
        }
        current_warp_idx = (current_warp_idx + 1) % warps.size();
    } while (current_warp_idx != start_idx);
    
    return nullptr;  // 没有活跃的warp
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

WarpContext* GreedyWarpScheduler::schedule_next() {
    // 首先尝试从ready队列中获取warp
    if (!ready_warps.empty()) {
        WarpContext* warp = ready_warps.front();
        ready_warps.pop();
        if (warp && warp->is_active() && !warp->is_finished()) {
            return warp;
        }
        // 如果warp不再活跃，继续寻找
    }
    
    // 遍历所有warp找到下一个活跃的
    for (auto* warp : warps) {
        if (warp && warp->is_active() && !warp->is_finished()) {
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