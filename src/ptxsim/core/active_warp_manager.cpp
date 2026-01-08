#include "ptxsim/active_warp_manager.h"
#include "utils/logger.h"

ActiveWarpManager::ActiveWarpManager(size_t max_warps) 
    : current_index_(0) {
    active_warps_.reserve(max_warps);
}

void ActiveWarpManager::add_active_warp(WarpContext* warp) {
    if (!warp) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 检查warp是否已经存在于活跃列表中
    for (auto* active_warp : active_warps_) {
        if (active_warp == warp) {
            return; // 已存在，无需重复添加
        }
    }
    
    active_warps_.push_back(warp);
    PTX_DEBUG_EMU("Added warp %p to active list, total active warps: %zu", 
                  warp, active_warps_.size());
}

void ActiveWarpManager::remove_inactive_warp(WarpContext* warp) {
    if (!warp) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = active_warps_.begin();
    while (it != active_warps_.end()) {
        if (*it == warp) {
            it = active_warps_.erase(it);
            PTX_DEBUG_EMU("Removed warp %p from active list, total active warps: %zu", 
                          warp, active_warps_.size());
            return;
        } else {
            ++it;
        }
    }
}

WarpContext* ActiveWarpManager::get_next_warp() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (active_warps_.empty()) {
        return nullptr;
    }
    
    // 使用round-robin策略选择下一个warp
    WarpContext* next_warp = active_warps_[current_index_];
    current_index_ = (current_index_ + 1) % active_warps_.size();
    
    // 如果warp已完成，则继续查找下一个活跃的warp
    int start_index = current_index_;
    while (next_warp->is_finished()) {
        current_index_ = (current_index_ + 1) % active_warps_.size();
        next_warp = active_warps_[current_index_];
        
        // 如果已经循环一圈，说明所有warp都已完成
        if (current_index_ == start_index) {
            break;
        }
    }
    
    return next_warp;
}

void ActiveWarpManager::update_warp_status(WarpContext* warp) {
    if (!warp) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (warp->is_finished()) {
        // 如果warp已完成，从活跃列表中移除
        auto it = active_warps_.begin();
        while (it != active_warps_.end()) {
            if (*it == warp) {
                it = active_warps_.erase(it);
                PTX_DEBUG_EMU("Removed finished warp %p from active list", warp);
                break;
            } else {
                ++it;
            }
        }
    } else {
        // 如果warp不在活跃列表中，添加它
        bool found = false;
        for (auto* active_warp : active_warps_) {
            if (active_warp == warp) {
                found = true;
                break;
            }
        }
        
        if (!found) {
            active_warps_.push_back(warp);
            PTX_DEBUG_EMU("Added warp %p to active list, total active warps: %zu", 
                          warp, active_warps_.size());
        }
    }
}

bool ActiveWarpManager::all_warps_finished() const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto* warp : active_warps_) {
        if (!warp->is_finished()) {
            return false;
        }
    }
    return true;
}

size_t ActiveWarpManager::get_active_warp_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_warps_.size();
}