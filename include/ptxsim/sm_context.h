#ifndef SM_CONTEXT_H
#define SM_CONTEXT_H

#include "cta_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/interpreter.h"
#include "thread_context.h"
#include "warp_context.h"
#include "warp_scheduler.h"
#include <map>
#include <memory>
#include <vector>

class CTAContext;

class SMContext {
public:
    SMContext(int max_warps, int max_threads_per_sm, size_t shared_mem_size);
    virtual ~SMContext() = default;

    // 初始化SM上下文
    void init(Dim3 &gridDim, Dim3 &blockDim,
              std::vector<StatementContext> &statements,
              std::map<std::string, PtxInterpreter::Symtable *> &name2Sym,
              std::map<std::string, int> &label2pc);

    // 添加线程块到SM
    bool add_block(CTAContext *block);

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

    // 网格和块维度信息
    Dim3 gridDim;
    std::vector<CTAContext *> blocks;

    // 共享内存管理
    std::map<std::string, PtxInterpreter::Symtable *> shared_memory;
};

#endif // SM_CONTEXT_H