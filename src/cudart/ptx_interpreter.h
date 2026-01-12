#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "ptx_ir/ptx_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include <map>
#include <memory>
#include <string>

// 前向声明
class GPUContext;

// 声明全局GPUContext实例
extern std::unique_ptr<GPUContext> g_gpu_context;

class PtxInterpreter {
public:
    // 构造函数，不再创建GPUContext
    explicit PtxInterpreter();

    PtxContext *ptxContext;
    KernelContext *kernelContext;
    void **kernelArgs;
    Dim3 gridDim{1, 1, 1}, blockDim{1, 1, 1};

    std::map<std::string, uint64_t> constName2addr;

    // PARAM空间管理
    void *param_space;

    void launchPtxInterpreter(PtxContext &ptx, std::string &kernel, void **args,
                              Dim3 &gridDim, Dim3 &blockDim);

    // 内部方法，接收必要的参数用于构建KernelLaunchRequest
    void funcInterpreter(std::map<std::string, Symtable *> &name2Sym,
                         std::map<std::string, int> &label2pc, PtxContext &ptx,
                         std::string &kernel, void **args, Dim3 &gridDim,
                         Dim3 &blockDim);
    void set_ptx_context(PtxContext &ptx);
    PtxContext &get_ptx_context();

private:
    void setupConstantSymbols(std::map<std::string, Symtable *> &name2Sym);
    void setupKernelArguments(std::map<std::string, Symtable *> &name2Sym);
    void setupLabels(std::map<std::string, int> &label2pc);
};

#endif // INTERPRETER_H