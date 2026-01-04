#ifndef WARP_CONTEXT_H
#define WARP_CONTEXT_H

#include "ptxsim/thread_context.h"
#include "ptx_ir/statement_context.h"
#include <vector>
#include <array>

class WarpContext {
public:
    static constexpr int WARP_SIZE = 32;
    
    WarpContext();
    virtual ~WarpContext() = default;
    
    // 添加线程到warp
    void add_thread(ThreadContext* thread, int lane_id);
    
    // 执行warp的一条指令
    void execute_warp_instruction(StatementContext& stmt);
    
    // 获取warp中的线程
    ThreadContext* get_thread(int lane_id) {
        if (lane_id >= 0 && lane_id < threads.size()) {
            return threads[lane_id];
        }
        return nullptr;
    }
    
    // 检查warp是否活跃
    bool is_active() const { return active_count > 0; }
    
    // 获取活跃线程数量
    int get_active_count() const { return active_count; }
    
    // 获取PC值
    int get_pc() const { return pc; }
    
    // 设置PC值
    void set_pc(int new_pc) { pc = new_pc; }
    
    // 更新活跃掩码（例如，遇到分支指令时）
    void update_active_mask();
    
    // 设置活跃掩码
    void set_active_mask(int lane_id, bool active);
    
    // 检查特定lane是否活跃
    bool is_lane_active(int lane_id) const {
        return lane_id >= 0 && lane_id < WARP_SIZE && active_mask[lane_id];
    }
    
    // 获取warp内线程ID
    int get_warp_thread_id(int lane_id) const {
        return lane_id < WARP_SIZE ? warp_thread_ids[lane_id] : -1;
    }
    
    // 获取warp索引
    int get_warp_id() const { return warp_id; }
    
    // 设置warp索引
    void set_warp_id(int id) { warp_id = id; }
    
    // 重置warp状态
    void reset();
    
    // 检查warp是否完成
    bool is_finished() const;
    
    // 同步warp内所有线程
    void sync_threads();
    
    // 处理分支分歧
    void handle_branch_divergence(int lane_id, int new_pc);
    
    // 检查是否有分歧
    bool has_divergence() const { return divergence_detected; }
    
    // 获取活跃掩码（32位）
    uint32_t get_active_mask() const;
    
    // 设置活跃掩码（32位）
    void set_active_mask(uint32_t mask);

private:
    std::vector<ThreadContext*> threads;  // warp中的线程指针
    std::array<bool, WARP_SIZE> active_mask;  // 活跃掩码
    std::array<int, WARP_SIZE> warp_thread_ids;  // 对应的线程ID
    int active_count;  // 活跃线程数量
    int pc;  // warp级PC
    int warp_id;  // warp ID
    bool single_step_mode;  // 单步执行模式
    bool divergence_detected;  // 是否检测到分歧
    std::vector<int> pc_stacks[WARP_SIZE];  // 每个线程的PC栈，用于分支重新合并
};

#endif // WARP_CONTEXT_H