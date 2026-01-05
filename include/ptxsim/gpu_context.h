#ifndef GPU_CONTEXT_H
#define GPU_CONTEXT_H

#include "cta_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h" // 包含通用类型定义
#include "ptxsim/execution_types.h"
#include "ptxsim/interpreter.h"
#include "sm_context.h"
#include "warp_context.h"
#include <condition_variable>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

// GPU硬件配置结构体
struct GPUConfig {
    int num_sms;
    int max_warps_per_sm;
    int max_threads_per_sm;
    size_t shared_mem_size_per_sm;
    int registers_per_sm;
    int max_blocks_per_sm;
    int warp_size;

    // 构造函数提供默认值
    GPUConfig()
        : num_sms(80),              // 默认80个SM，类似Ampere A100
          max_warps_per_sm(64),     // 每个SM最大64个warp
          max_threads_per_sm(2048), // 每个SM最大2048个线程
          shared_mem_size_per_sm(1024 * 64), // 每个SM 64KB共享内存
          registers_per_sm(65536),           // 每个SM 64K寄存器
          max_blocks_per_sm(32),             // 每个SM最大32个block
          warp_size(32)                      // warp大小为32
    {}
};

// 内核启动请求结构体
struct KernelLaunchRequest {
    void **args;
    Dim3 gridDim;
    Dim3 blockDim;
    std::vector<StatementContext> *statements;
    std::map<std::string, Symtable *> *name2Sym;
    std::map<std::string, int> *label2pc;

    KernelLaunchRequest(void **_args, Dim3 &_gridDim, Dim3 &_blockDim,
                        std::vector<StatementContext> *_stmts,
                        std::map<std::string, Symtable *> *_name2Sym,
                        std::map<std::string, int> *_label2pc)
        : args(_args), gridDim(_gridDim), blockDim(_blockDim),
          statements(_stmts), name2Sym(_name2Sym), label2pc(_label2pc) {}
};

class GPUContext {
public:
    explicit GPUContext(const std::string &config_path = "");
    virtual ~GPUContext() = default;

    // 从配置文件加载GPU配置
    bool load_config(const std::string &config_path);

    // 初始化GPU上下文
    void init(Dim3 &gridDim, Dim3 &blockDim,
              std::vector<StatementContext> &statements,
              std::map<std::string, Symtable *> &name2Sym,
              std::map<std::string, int> &label2pc);

    // 同步执行kernel
    bool launch_kernel(void **args, Dim3 &gridDim, Dim3 &blockDim,
                       std::vector<StatementContext> &statements,
                       std::map<std::string, Symtable *> &name2Sym,
                       std::map<std::string, int> &label2pc);

    // 异步执行kernel
    std::future<EXE_STATE>
    launch_kernel_async(void **args, Dim3 &gridDim, Dim3 &blockDim,
                        std::vector<StatementContext> &statements,
                        std::map<std::string, Symtable *> &name2Sym,
                        std::map<std::string, int> &label2pc);

    // 执行一个GPU周期
    EXE_STATE exe_once();

    // 获取GPU状态
    EXE_STATE get_state() const { return gpu_state; }

    // 获取SM数量
    size_t get_num_sms() const { return sms.size(); }

    // 获取指定SM
    SMContext *get_sm(size_t idx) {
        return idx < sms.size() ? sms[idx].get() : nullptr;
    }

    // 获取GPU配置
    const GPUConfig &get_config() const { return config; }

    // 检查是否有待处理的任务
    bool has_pending_tasks() const;

private:
    // GPU配置
    GPUConfig config;

    // SM列表
    std::vector<std::unique_ptr<SMContext>> sms;

    // GPU状态
    EXE_STATE gpu_state;

    // 任务队列
    std::queue<KernelLaunchRequest> task_queue;
    mutable std::mutex queue_mutex;
    std::condition_variable task_cv;

    // 当前执行的CTA列表
    std::vector<std::unique_ptr<CTAContext>> active_ctas;

    // 网格和块维度信息（用于初始化）
    Dim3 gridDim;
    Dim3 blockDim;

    // 存储语句和符号表
    std::vector<StatementContext> *statements;
    std::map<std::string, Symtable *> *name2Sym;
    std::map<std::string, int> *label2pc;

    // 内部执行kernel的辅助函数
    bool execute_kernel_internal(void **args, Dim3 &gridDim, Dim3 &blockDim,
                                 std::vector<StatementContext> &statements,
                                 std::map<std::string, Symtable *> &name2Sym,
                                 std::map<std::string, int> &label2pc);
};

#endif // GPU_CONTEXT_H