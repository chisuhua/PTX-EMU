#ifndef CTA_CONTEXT_H
#define CTA_CONTEXT_H

#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h" // 包含通用类型定义
#include "ptxsim/execution_types.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include <map>
#include <memory>
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
    std::map<std::string, Symtable *> name2Local; // 本地内存符号表
    std::map<std::string, Symtable *> name2Share; // 本地内存符号表

    void init(Dim3 &GridDim, Dim3 &BlockDim, Dim3 &blockIdx,
              std::vector<StatementContext> &statements,
              std::map<std::string, Symtable *> *name2Sym,
              std::map<std::string, int> &label2pc,
              void *local_memory_base = nullptr,
              size_t local_mem_per_thread = 0);

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
};

#endif // CTA_CONTEXT_H