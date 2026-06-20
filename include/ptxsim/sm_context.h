#ifndef SM_CONTEXT_H
#define SM_CONTEXT_H

#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_scheduler.h"
#include <deque>
#include <map>
#include <memory>
#include <set> // 添加set头文件
#include <vector>

class WarpScheduler;
class CTAContext;
class SharedMemoryManager;
class ThreadContext; // 添加ThreadContext前向声明

class SMContext {
public:
    SMContext(int max_warps, int max_threads_per_sm, size_t shared_mem_size,
              int sm_id);
    virtual ~SMContext();

    // 初始化SM上下文，不再需要任务相关参数
    void init();

    // 添加块到SM，接收unique_ptr以转移所有权
    bool add_block(std::unique_ptr<CTAContext> block);

    // 执行一个SM周期
    EXE_STATE exe_once();

    // 获取SM状态
    EXE_STATE get_state() const { return sm_state; }

    // 检查是否空闲
    bool is_idle() const;

    // 获取活跃warp数量
    int get_active_warps_count() const;

    // 获取活跃线程数量
    int get_active_threads_count() const;

    // 获取已分配的共享内存大小
    size_t get_allocated_shared_mem() const { return allocated_shared_mem; }

    // 获取最大共享内存大小
    size_t get_max_shared_mem() const { return max_shared_mem; }

    // 获取warp调度器
    WarpScheduler *get_warp_scheduler() { return warp_scheduler.get(); }

    // 设置warp调度器策略
    void set_warp_scheduler(std::unique_ptr<WarpScheduler> scheduler);

    // 获取当前活跃的warp数量
    size_t get_num_warps() const { return warps.size(); }

    // 获取当前周期计数
    uint64_t get_cycle_count() const { return cycle_counter_; }
    int get_sm_id() const { return sm_id_; }

    // 获取WarpContext
    WarpContext *get_warp(size_t idx) {
        return idx < warps.size() ? warps[idx].get() : nullptr;
    }

    // 清理已完成的块
    void cleanup_finished_blocks();

    // 从 pending_blocks_ 队列尽可能多地 admit 新 block
    // 由 cleanup_finished_blocks() 和 add_block() 在资源释放后自动调用
    // 公开此方法以便测试和 GPUContext::execute_kernel_internal 显式触发
    void try_admit_pending_blocks();

    // 预留资源
    bool reserve_resources(size_t shared_mem_size, int warp_count);

    // 释放资源
    void release_resources(int reservation_id);

    // BUG-SM-ADMISSION-OVERFLOW: streaming admission 观察接口
    // get_admitted_block_count() == managed_blocks.size()
    // get_pending_block_count()  == pending_blocks_.size()
    // get_total_block_count()    == 上述两者之和
    // 不变式: add_block 成功调用后, total 必然 +1,绝不静默丢块
    size_t get_admitted_block_count() const { return managed_blocks.size(); }
    size_t get_pending_block_count() const { return pending_blocks_.size(); }
    size_t get_total_block_count() const {
        return managed_blocks.size() + pending_blocks_.size();
    }

    // 获取资源使用统计
    struct ResourceStats {
        size_t allocated_shared_mem;
        size_t max_shared_mem;
        int active_warps;
        int max_warps;
        int active_threads;
        int max_threads;
    };

    ResourceStats get_resource_stats() const;

    // 打印资源使用情况
    void print_resource_usage() const;

    // 新增：调试打印函数，用于显示warp状态信息
    void print_warp_status() const;
    void print_warp_status(const WarpContext *warp,
                           bool print_sm_id = true) const;

    int select_next_group(const std::vector<int>& active_lanes);
    void suspend_and_switch(int current_group, int next_group);

private:
    // 初始化warp
    void init_warps_for_block(CTAContext *block);

    // 更新SM状态
    void update_state();

    // 分配共享内存
    bool allocate_shared_memory(CTAContext *block);

    // 释放共享内存
    void free_shared_memory(CTAContext *block);

    // 最大资源限制
    int max_warps_per_sm;
    int max_threads_per_sm;
    size_t max_shared_mem;

    // 当前资源使用情况
    size_t allocated_shared_mem;
    int current_thread_count;

    // SM状态
    EXE_STATE sm_state;

    // Warp相关
    std::vector<std::unique_ptr<WarpContext>> warps;
    std::unique_ptr<WarpScheduler> warp_scheduler;

    // 使用unique_ptr管理CTAContext的生命周期
    std::map<int, std::unique_ptr<CTAContext>> managed_blocks;

    // BUG-SM-ADMISSION-OVERFLOW: blocks 等待 admit 队列(FIFO)
    // 当 add_block 因资源不足失败时,块进 pending_blocks_
    // cleanup_finished_blocks() 后, try_admit_pending_blocks() 自动重灌
    // 不变式: managed_blocks.size() + pending_blocks_.size() ==
    //          累计成功 add_block 调用次数
    std::deque<std::unique_ptr<CTAContext>> pending_blocks_;

    // 记录每个块的warp总数和已完成warp数（使用物理ID）
    std::map<int, int> physical_block_warp_counts;

    // 物理ID生成器
    int next_physical_block_id = 0;
    int next_physical_warp_id = 0;

    // 共享内存管理
    std::map<std::string, Symtable *> shared_memory;

    // 资源管理器引用
    SharedMemoryManager *shared_mem_manager_ = nullptr;

    // 资源预留ID
    int current_reservation_id_ = 0;

    // 资源统计
    mutable ResourceStats stats_;

    // SM ID
    int sm_id_;

    // 周期计数器（每执行一次 exe_once 递增）
    uint64_t cycle_counter_;

    friend class BarWarpSyncHandler;

    // Divergence execution mode
    ptxsim::DivergenceExecutionMode divergence_mode_ = ptxsim::DivergenceExecutionMode::Sequential;

public:
    // Divergence execution mode methods
    void set_divergence_execution_mode(ptxsim::DivergenceExecutionMode mode);
    ptxsim::DivergenceExecutionMode get_divergence_execution_mode() const;
};

#endif // SM_CONTEXT_H
