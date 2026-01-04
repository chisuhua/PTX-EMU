#include "ptxsim/interpreter.h"
#include "memory/memory_manager.h" // 添加 MemoryManager 头文件
#include "ptx_ir/kernel_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/sm_context.h"
#include "utils/logger.h"
#include <cassert>
#include <cstring>
#include <map>

void PtxInterpreter::launchPtxInterpreter(PtxContext &ptx, std::string &kernel,
                                          void **args, Dim3 &gridDim,
                                          Dim3 &blockDim) {
    // 初始化指令工厂，注册所有指令处理器
    InstructionFactory::initialize();

    this->ptxContext = &ptx;
    this->gridDim = gridDim;
    this->blockDim = blockDim;
    this->kernelArgs = args;
    this->param_space = nullptr; // 初始化param_space

    // 根据kernel名称获取kernelContext
    for (auto &e : ptx.ptxKernels) {
        if (e.kernelName == kernel) {
            this->kernelContext = &e;
            break;
        }
    }

    std::map<std::string, PtxInterpreter::Symtable *> name2Sym;
    std::map<std::string, int> label2pc;

    funcInterpreter(name2Sym, label2pc);

    // 内核执行结束后，释放PARAM空间，使用 MemoryManager 提供的 free_param 函数
    if (this->param_space) {
        PTX_DEBUG_EMU("Freeing PARAM space at %p", this->param_space);
        MemoryManager::instance().free_param(this->param_space);
        this->param_space = nullptr;
    }

    // 清理符号表
    for (auto &pair : name2Sym) {
        delete pair.second;
    }
}

void PtxInterpreter::funcInterpreter(
    std::map<std::string, Symtable *> &name2Sym,
    std::map<std::string, int> &label2pc) {
    // Setup symbols
    setupConstantSymbols(name2Sym);
    setupKernelArguments(name2Sym);
    setupLabels(label2pc);

    // 创建SM上下文，模拟硬件SM
    // 这里我们创建一个SM，实际中可能有多个SM
    SMContext sm(32, 2048,
                 1024 * 64); // 假设每个SM最多32个warp，2048个线程，64KB共享内存
    sm.init(gridDim, blockDim, kernelContext->kernelStatements, name2Sym,
            label2pc);

    int ctaNum = gridDim.x * gridDim.y * gridDim.z;

    // 为每个CTA创建上下文并添加到SM
    for (int i = 0; i < ctaNum; i++) {
        Dim3 blockIdx;
        blockIdx.z = i / (gridDim.x * gridDim.y);
        blockIdx.y = i % (gridDim.x * gridDim.y) / (gridDim.x);
        blockIdx.x = i % (gridDim.x * gridDim.y) % (gridDim.x);

        // 创建CTAContext
        CTAContext *cta = new CTAContext();
        cta->init(gridDim, blockDim, blockIdx, kernelContext->kernelStatements,
                  name2Sym, label2pc);

        // 将CTA添加到SM
        if (!sm.add_block(cta)) {
            // 如果SM资源不足，可以创建另一个SM或等待
            delete cta;
            break; // 简化处理：如果当前SM无法容纳更多块，则停止
        }
    }

    // 执行直到所有CTA完成
    while (sm.get_state() != EXIT) {
        sm.exe_once();
    }
}

void PtxInterpreter::setupConstantSymbols(
    std::map<std::string, Symtable *> &name2Sym) {
    for (auto e : ptxContext->ptxStatements) {
        assert(e.statementType == S_CONST);
        Symtable *s = new Symtable();
        auto st = (StatementContext::CONST *)e.statement;
        assert(st->constDataType.size() == 1);
        s->name = st->constName;
        s->symType = st->constDataType.back();
        s->elementNum = st->constSize;
        s->byteNum = Q2bytes(st->constDataType.back());
        s->val = constName2addr[s->name];
        assert(s->val);
        name2Sym[s->name] = s;
    }
}

void PtxInterpreter::setupKernelArguments(
    std::map<std::string, Symtable *> &name2Sym) {
    PTX_DEBUG_EMU("Setting up %zu kernel arguments",
                  kernelContext->kernelParams.size());

    // 计算参数总大小
    size_t total_param_size = 0;
    for (int i = 0; i < kernelContext->kernelParams.size(); i++) {
        auto e = kernelContext->kernelParams[i];
        total_param_size +=
            Q2bytes(e.paramType) * (e.paramNum ? e.paramNum : 1);
    }

    // 申请PARAM空间，使用 MemoryManager 提供的 malloc_param 函数
    if (total_param_size > 0) {
        this->param_space =
            MemoryManager::instance().malloc_param(total_param_size);
        if (this->param_space == nullptr) {
            PTX_DEBUG_EMU("Failed to allocate PARAM space of size %zu",
                          total_param_size);
            return; // 或者抛出异常
        }
        memset(this->param_space, 0, total_param_size);
        PTX_DEBUG_EMU("Allocated PARAM space of size %zu at %p",
                      total_param_size, this->param_space);
    } else {
        this->param_space = nullptr;
        PTX_DEBUG_EMU("No PARAM space needed, total_param_size is 0");
    }

    // 遍历参数，将值填入PARAM空间，并在符号表中记录地址
    size_t offset = 0;
    for (int i = 0; i < kernelContext->kernelParams.size(); i++) {
        auto e = kernelContext->kernelParams[i];
        Symtable *s = new Symtable();
        s->name = e.paramName;
        s->elementNum = e.paramNum;
        s->symType = e.paramType;
        s->byteNum = Q2bytes(e.paramType);

        // 计算当前参数大小
        size_t param_size = s->byteNum * (e.paramNum ? e.paramNum : 1);

        // 检查是否需要分配空间
        if (this->param_space != nullptr) {
            // 将参数值拷贝到PARAM空间
            memcpy((char *)this->param_space + offset, kernelArgs[i],
                   param_size);
        }

        // 在符号表中存储PARAM空间中该参数的地址
        s->val = (uint64_t)((char *)this->param_space + offset);

        PTX_DEBUG_EMU(
            "Kernel argument[%d]: name=%s, elementNum=%d, byteNum=%d, "
            "param_size=%zu, param_space_offset=%zu, stored_addr=%p, "
            "first_8_bytes_of_data=0x%lx",
            i, s->name.c_str(), s->elementNum, s->byteNum, param_size, offset,
            (void *)s->val, *(uint64_t *)kernelArgs[i]);

        name2Sym[s->name] = s;
        offset += param_size;
    }
}

void PtxInterpreter::setupLabels(std::map<std::string, int> &label2pc) {
    for (int i = 0; i < kernelContext->kernelStatements.size(); i++) {
        auto e = kernelContext->kernelStatements[i];
        if (e.statementType == S_DOLLOR) {
            auto s = (StatementContext::DOLLOR *)e.statement;
            label2pc[s->dollorName] = i;
        }
    }
}