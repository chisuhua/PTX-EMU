#ifndef CTA_CONTEXT_H
#define CTA_CONTEXT_H

#include "ptx_ir/statement_context.h"
#include "ptxsim/barrier/barrier_module.h"
#include "ptxsim/common_types.h" // 包含通用类型定义
#include "ptxsim/execution_types.h"
#include "ptxsim/memory/tma_descriptor.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/cluster/cluster_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include <map>
#include <memory>
#include <optional>
#include <vector>

enum class CTAState {
    INIT,        // 初始化完成
    TRANSFERRED, // warp已转移至SM
    EXECUTING,   // 正在执行
    FINISHED     // 执行完成
};

class PtxInterpreter; // 前向声明

class CTAContext {
public:
    int warpNum;
    int curExeWarpId;

    int threadNum;
    int curExeThreadId;
    int exitThreadNum;
    int barThreadNum;

    size_t sharedMemBytes = 0;
    void *sharedMemSpace = nullptr; // 共享内存空间指针
    Dim3 blockIdx, BlockDim, GridDim;

    // 本地内存相关变量
    size_t localMemBytesPerThread = 0; // 每个线程的本地内存大小
    std::vector<void *> localMemSpaces; // 每个线程的本地内存空间指针
    std::map<std::string, std::unique_ptr<Symtable>> name2Local; // 本地内存符号表
    std::map<std::string, std::unique_ptr<Symtable>> name2Share; // 本地内存符号表

    void init(Dim3 &GridDim, Dim3 &BlockDim, Dim3 &blockIdx,
              std::vector<StatementContext> &statements,
              std::map<std::string, std::unique_ptr<Symtable>> *name2Sym,
              std::map<std::string, int> &label2pc,
              void *local_memory_base = nullptr,
              size_t local_mem_per_thread = 0,
              size_t dynamic_shared_mem_size = 0);

    // 新增方法：构建共享内存符号表，接收分配好的共享内存空间
    void build_shared_memory_symbol_table(void *shared_mem_space);

    // 新增方法：构建本地内存符号表，为每个线程分配本地内存空间
    void build_local_memory_symbol_table();

    EXE_STATE exe_once();

    bool allThreadsExited() const { return exitThreadNum == threadNum; }
    bool allThreadsAtBarrier() const { return barThreadNum == threadNum; }

    // 获取共享内存需求
    size_t get_shared_memory_requirement() const { return sharedMemBytes; }

    // 获取warp数量需求
    int get_warp_count() const { return warpNum; }
    WarpContext* get_warp(int warp_id) const;

    // 获取线程数量
    int get_thread_count() const { return threadNum; }

    // 获取和设置状态
    CTAState get_state() const { return state_; }
    void set_state(CTAState state) { state_ = state; }

    // 资源预留ID的getter和setter方法
    int get_reservation_id() const { return reservation_id_; }
    void set_reservation_id(int id) { reservation_id_ = id; }

    // 释放warp的所有权
    std::vector<std::unique_ptr<WarpContext>> release_warps();

    // CTA 级 barrier 管理（per-CTA 一个 BarrierModule）
    // Note: BarrierModule lives in ptxsim namespace (barrier/barrier_module.h).
    ptxsim::BarrierModule& get_barrier_module() { return *barrier_module_; }
    const ptxsim::BarrierModule& get_barrier_module() const { return *barrier_module_; }

    // Ensure barrier_module_ is allocated (for tests that construct
    // CTAContext manually without calling the full init() flow).
    void ensure_barrier_module() {
        if (!barrier_module_) {
            barrier_module_ = std::make_unique<ptxsim::BarrierModule>();
        }
    }

    // Phase 0.5.1 (Fix #9a): per-CTA TMA descriptor store accessor
    TmaDescriptorStore& tma_descriptor_store() { return tma_descriptor_store_; }
    const TmaDescriptorStore& tma_descriptor_store() const { return tma_descriptor_store_; }

    // Phase 0.5.2 (Fix #9b): per-CTA TMEM accessor
    Tmem& tmem() { return tmem_; }
    const Tmem& tmem() const { return tmem_; }

    // Phase 0.5.3 (Fix #9c): per-CTA cluster context (lazy-init via
    // std::optional — ClusterContext has explicit ctor, unlike
    // TmaDescriptorStore/Tmem which have default ctors).
    // Pre-condition: init_cluster_context() must be called before
    // cluster_context() accessor. has_cluster_context() for explicit check.
    void init_cluster_context(ClusterContext::cta_id_t root_id,
                              ClusterContext::cluster_size_t num_ctas) {
        cluster_context_.emplace(root_id, num_ctas);
    }
    ClusterContext& cluster_context() { return cluster_context_.value(); }
    const ClusterContext& cluster_context() const {
        return cluster_context_.value();
    }
    bool has_cluster_context() const { return cluster_context_.has_value(); }

    ~CTAContext();

private:
    // 存储初始化时的statements引用，用于后续构建共享内存符号表
    std::vector<StatementContext> *init_statements;

    // 状态管理
    CTAState state_ = CTAState::INIT;

    // 添加资源预留ID
    int reservation_id_ = -1;

    // 存储warp的向量，用于转移所有权
    std::vector<std::unique_ptr<WarpContext>> warps;

    // 统一管理 CTA 内 16 个 named barrier（NVIDIA 硬件对齐）
    std::unique_ptr<ptxsim::BarrierModule> barrier_module_;

    // Phase 0.5.1 (Fix #9a): per-CTA TmaDescriptorStore
    TmaDescriptorStore tma_descriptor_store_;

    // Phase 0.5.2 (Fix #9b): per-CTA TMEM
    Tmem tmem_;

    // Phase 0.5.3 (Fix #9c): per-CTA cluster context (lazy-init via
    // std::optional — ClusterContext has explicit ctor unlike default-ctored
    // TmaDescriptorStore/Tmem. emplace() constructs in-place; no move/copy.)
    std::optional<ClusterContext> cluster_context_;
};

#endif // CTA_CONTEXT_H