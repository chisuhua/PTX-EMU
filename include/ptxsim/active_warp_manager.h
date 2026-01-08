#ifndef ACTIVE_WARP_MANAGER_H
#define ACTIVE_WARP_MANAGER_H

#include "warp_context.h"
#include <vector>
#include <mutex>

class ActiveWarpManager {
public:
    explicit ActiveWarpManager(size_t max_warps);
    
    // 添加活跃warp
    void add_active_warp(WarpContext* warp);
    
    // 移除非活跃warp
    void remove_inactive_warp(WarpContext* warp);
    
    // 获取下一个要调度的warp
    WarpContext* get_next_warp();
    
    // 更新warp活跃状态
    void update_warp_status(WarpContext* warp);
    
    // 检查是否所有warp已完成
    bool all_warps_finished() const;
    
    // 获取活跃warp数量
    size_t get_active_warp_count() const;
    
private:
    std::vector<WarpContext*> active_warps_;
    size_t current_index_;
    mutable std::mutex mutex_;
};

#endif // ACTIVE_WARP_MANAGER_H