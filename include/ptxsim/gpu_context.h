#ifndef GPU_CONTEXT_H
#define GPU_CONTEXT_H

#include "cta_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h" // 包含通用类型定义
#include "ptxsim/execution_types.h"
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
        : num_sms(1),               // 默认80个SM，类似Ampere A100
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
    std::vector<StatementContext> *statements; // 直接引用，由ptxContext持有
    std::shared_ptr<std::map<std::string, Symtable *>> name2Sym;
    std::shared_ptr<std::map<std::string, int>> label2pc;
    int request_id; // 添加请求ID以追踪任务状态
    std::function<void()> on_complete; // 任务完成时的回调函数

    // 默认构造函数
    KernelLaunchRequest() : args(nullptr), gridDim(), blockDim(),
                            statements(nullptr), name2Sym(nullptr), label2pc(nullptr),
                            request_id(0), on_complete(nullptr) {}

    KernelLaunchRequest(
        void **_args, Dim3 &_gridDim, Dim3 &_blockDim,
        std::vector<StatementContext> *_stmts,
        std::shared_ptr<std::map<std::string, Symtable *>> _name2Sym,
        std::shared_ptr<std::map<std::string, int>> _label2pc,
        int _request_id = 0, std::function<void()> _on_complete = nullptr)
        : args(_args), gridDim(_gridDim), blockDim(_blockDim),
          statements(_stmts), name2Sym(_name2Sym), label2pc(_label2pc),
          request_id(_request_id), on_complete(_on_complete) {}
};

class GPUContext {
public:
    explicit GPUContext(const std::string &config_path = "");
    virtual ~GPUContext() = default;

    // 从配置文件加载GPU配置
    bool load_json_config(const std::string &config_path);

    // 硬件初始化，不再需要任务参数
    void init();

    // 提交kernel请求
    void submit_kernel_request(KernelLaunchRequest &&request);

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

    // 检查是否有等待的任务
    bool has_pending_tasks() const;

    // 等待所有任务完成
    void wait_for_completion();

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

    // 正在执行的任务映射
    std::map<int, KernelLaunchRequest> executing_requests;
    int next_request_id = 0;

    // 内部执行kernel的辅助函数
    bool execute_kernel_internal(void **args, Dim3 &gridDim, Dim3 &blockDim,
                                 std::vector<StatementContext> &statements,
                                 std::map<std::string, Symtable *> &name2Sym,
                                 std::map<std::string, int> &label2pc);
};

#endif // GPU_CONTEXT_H