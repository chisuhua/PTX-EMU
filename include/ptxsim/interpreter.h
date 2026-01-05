#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "ptx_ir/ptx_context.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/qualifier.h"
#include "execution_types.h"
#include "gpu_context.h"
#include <map>
#include <string>
#include <memory>

class PtxInterpreter {
public:
    class Symtable { // integrate param local const
    public:
        Qualifier symType;
        int byteNum;
        int elementNum;
        std::string name;
        uint64_t val;
    };

    class Reg {
    public:
        Qualifier regType;
        int byteNum;
        int elementNum;
        std::string name;
        void *addr;
    };

    class IMM {
    public:
        Qualifier type;
        union Data {
            uint8_t u8;
            uint16_t u16;
            uint32_t u32;
            uint64_t u64;
            float f32;
            double f64;
        };
        Data data;
    };

    class VEC {
    public:
        std::vector<void *> vec;
    };

public:
    // 构造函数，可以选择是否使用配置文件
    explicit PtxInterpreter(const std::string& gpu_config_path = "");

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