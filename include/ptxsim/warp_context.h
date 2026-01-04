#ifndef WARP_CONTEXT_H
#define WARP_CONTEXT_H

#include "ptxsim/thread_context.h"
#include "ptx_ir/statement_context.h"
#include <vector>
#include <memory>

class WarpContext {
public:
    static const int WARP_SIZE = 32;
    
    // warp 内的线程 ID 映射
    int warp_thread_ids[WARP_SIZE];
    bool active_mask[WARP_SIZE];  // 活跃线程掩码
    int active_count;             // 活跃线程数
    
    // warp 状态
    uint32_t pc;                  // warp 的程序计数器
    bool single_step_mode;        // 单步执行模式
    
    // 执行控制
    std::vector<ThreadContext*> threads;
    
    // 构造函数
    WarpContext();
    
    // 执行单个 warp 指令
    void execute_warp_instruction(StatementContext &stmt);
    
    // 更新活跃掩码
    void update_active_mask();
    
    // 检查 warp 是否全部完成
    bool is_complete() const;
    
    // 同步 warp 内所有线程
    void sync_warp();
    
    // 添加线程到warp
    void add_thread(ThreadContext* thread, int lane_id);
    
    // 获取活跃线程数
    int get_active_count() const;
    
    // 检查是否有活跃线程
    bool has_active_threads() const;
    
    // 获取warp的活跃掩码
    uint32_t get_active_mask() const;
    
    // 设置活跃掩码
    void set_active_mask(uint32_t mask);
    
    // 设置指定lane的活跃状态
    void set_lane_active(int lane_id, bool active);
    
    // 检查指定lane是否活跃
    bool is_lane_active(int lane_id) const;
};

#endif // WARP_CONTEXT_H