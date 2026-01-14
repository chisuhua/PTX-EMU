#ifndef WARP_CONTEXT_H
#define WARP_CONTEXT_H

#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "register/register_bank_manager.h"
#include <array>
#include <memory>
#include <queue>
#include <vector>

// Forward declarations to avoid circular includes
class SMContext;
class ThreadContext;  // 添加ThreadContext的前向声明

class WarpContext {
public:
    static constexpr int WARP_SIZE = 32;

    WarpContext();
    virtual ~WarpContext() = default;

    // 添加线程到warp
    void add_thread(std::unique_ptr<ThreadContext> thread, int lane_id);

    // 执行warp的一条指令
    void execute_warp_instruction(StatementContext &stmt);

    // 获取warp中的线程
    ThreadContext *get_thread(int lane_id) const {
        if (lane_id >= 0 && lane_id < threads.size()) {
            return threads[lane_id].get();
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

    // 检查warp是否完成 - 现在检查是否所有线程都已退出
    bool is_finished() const;

    // 检查warp是否真正完成（所有线程都已退出），而不是仅活跃计数为0
    bool is_all_threads_exited() const;

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

    // 设置寄存器银行管理器
    void
    set_register_bank_manager(std::shared_ptr<RegisterBankManager> manager) {
        register_bank_manager_ = manager;
    }

    // 获取warp中所有线程的引用
    const std::vector<std::unique_ptr<ThreadContext>>& get_threads() const {
        return threads;
    }

    // 获取指定范围内的活跃线程
    std::vector<ThreadContext*> get_active_threads() const {
        std::vector<ThreadContext*> active_threads;
        for (int i = 0; i < threads.size(); ++i) {
            if (is_lane_active(i) && threads[i]) {
                active_threads.push_back(threads[i].get());
            }
        }
        return active_threads;
    }

    // 设置SM Context
    void set_sm_context(SMContext *sm_ctx) { sm_context_ = sm_ctx; }
    
    // 获取SM Context
    SMContext *get_sm_context() const { return sm_context_; }

private:
    std::vector<std::unique_ptr<ThreadContext>>
        threads;                                // warp中的线程unique_ptr
    std::array<bool, WARP_SIZE> active_mask;    // 活跃掩码
    std::array<int, WARP_SIZE> warp_thread_ids; // 对应的线程ID
    int active_count;                           // 活跃线程数量
    int pc;                                     // warp级PC
    int warp_id;                                // warp ID
    int physical_warp_id;                       // 物理warp ID
    int physical_block_id;                      // 物理warp ID

    bool divergence_detected;              // 分歧检测标志
    std::vector<int> pc_stacks[WARP_SIZE]; // 每个线程的PC栈，用于分支重新合并

    // 寄存器银行管理器
    std::shared_ptr<RegisterBankManager> register_bank_manager_;

    // 单步执行模式
    bool single_step_mode;

    // 指向SMContext的指针
    SMContext *sm_context_ = nullptr;

    // 调度状态
    bool is_scheduled_{false}; // 表示warp是否被调度执行

public:
    // 调度状态相关方法
    void set_scheduled(bool scheduled) { is_scheduled_ = scheduled; }
    bool is_scheduled() const { return is_scheduled_; }

    // 物理ID管理方法
    void set_physical_warp_id(int id) { physical_warp_id = id; }
    int get_physical_warp_id() const { return physical_warp_id; }

    void set_physical_block_id(int id) { physical_block_id = id; }
    int get_physical_block_id() const { return physical_block_id; }
};

#endif // WARP_CONTEXT_H