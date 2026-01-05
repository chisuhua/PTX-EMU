#ifndef GPU_CONTEXT_H
#define GPU_CONTEXT_H

#include "sm_context.h"
#include "cta_context.h"
#include "warp_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/interpreter.h"
#include <vector>
#include <memory>
#include <map>
#include <string>

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
    GPUConfig() : 
        num_sms(80),           // 默认80个SM，类似Ampere A100
        max_warps_per_sm(64),  // 每个SM最大64个warp
        max_threads_per_sm(2048), // 每个SM最大2048个线程
        shared_mem_size_per_sm(1024 * 64), // 每个SM 64KB共享内存
        registers_per_sm(65536), // 每个SM 64K寄存器
        max_blocks_per_sm(32),   // 每个SM最大32个block
        warp_size(32)            // warp大小为32
    {}
};

class GPUContext {
public:
    explicit GPUContext(const std::string& config_path = "");
    virtual ~GPUContext() = default;
    
    // 从配置文件加载GPU配置
    bool load_config(const std::string& config_path);
    
    // 初始化GPU上下文
    void init(Dim3& gridDim, Dim3& blockDim,
              std::vector<StatementContext>& statements,
              std::map<std::string, PtxInterpreter::Symtable*>& name2Sym,
              std::map<std::string, int>& label2pc);
    
    // 添加kernel执行任务
    bool launch_kernel(void** args, Dim3& gridDim, Dim3& blockDim);
    
    // 执行一个GPU周期
    EXE_STATE exe_once();
    
    // 获取GPU状态
    EXE_STATE get_state() const { return gpu_state; }
    
    // 获取SM数量
    size_t get_num_sms() const { return sms.size(); }
    
    // 获取指定SM
    SMContext* get_sm(size_t idx) { 
        return idx < sms.size() ? sms[idx].get() : nullptr; 
    }
    
    // 获取GPU配置
    const GPUConfig& get_config() const { return config; }

private:
    // GPU配置
    GPUConfig config;
    
    // SM列表
    std::vector<std::unique_ptr<SMContext>> sms;
    
    // GPU状态
    EXE_STATE gpu_state;
    
    // 网格和块维度信息
    Dim3 gridDim;
    Dim3 blockDim;
    
    // 存储语句和符号表
    std::vector<StatementContext>* statements;
    std::map<std::string, PtxInterpreter::Symtable*>* name2Sym;
    std::map<std::string, int>* label2pc;
};

#endif // GPU_CONTEXT_H