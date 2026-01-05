#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "gpu_context.h"
#include "ptx_ir/ptx_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/gpu_context.h"
#include <map>
#include <memory>
#include <string>

class GPUContext;

class PtxInterpreter {
public:
    // 构造函数，可以选择是否使用配置文件
    explicit PtxInterpreter(const std::string &gpu_config_path = "");

    PtxContext *ptxContext;
    KernelContext *kernelContext;
    void **kernelArgs;
    Dim3 gridDim{1, 1, 1}, blockDim{1, 1, 1};

    std::map<std::string, uint64_t> constName2addr;

    // PARAM空间管理
    void *param_space;

    // GPU上下文，用于管理硬件资源
    std::shared_ptr<GPUContext> gpu_context;

    void launchPtxInterpreter(PtxContext &ptx, std::string &kernel, void **args,
                              Dim3 &gridDim, Dim3 &blockDim);

    void funcInterpreter(std::map<std::string, Symtable *> &name2Sym,
                         std::map<std::string, int> &label2pc);

private:
    void setupConstantSymbols(std::map<std::string, Symtable *> &name2Sym);
    void setupKernelArguments(std::map<std::string, Symtable *> &name2Sym);
    void setupLabels(std::map<std::string, int> &label2pc);
};

#endif // INTERPRETER_H