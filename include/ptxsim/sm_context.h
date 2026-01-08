#ifndef SM_CONTEXT_H
#define SM_CONTEXT_H

#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_scheduler.h"
#include <map>
#include <memory>
#include <vector>

class WarpScheduler;
class CTAContext;
class SharedMemoryManager;

class SMContext {
public:
    SMContext(int max_warps, int max_threads_per_sm, size_t shared_mem_size, int sm_id);
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

    // 获取WarpContext
    WarpContext *get_warp(size_t idx) {
        return idx < warps.size() ? warps[idx].get() : nullptr;
    }

    // 清理已完成的块
    void cleanup_finished_blocks();

    // 预留资源
    bool reserve_resources(size_t shared_mem_size, int warp_count);

    // 释放资源
    void release_resources(int reservation_id);

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
};

#endif // SM_CONTEXT_H