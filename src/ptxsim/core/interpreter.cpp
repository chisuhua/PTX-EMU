#include "ptxsim/interpreter.h"
#include "ptx_ir/kernel_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include <cassert>
#include <map>

void PtxInterpreter::launchPtxInterpreter(PtxContext &ptx, std::string &kernel,
                                          void **args, dim3 &gridDim,
                                          dim3 &blockDim) {
    this->ptxContext = &ptx;
    this->gridDim = gridDim;
    this->blockDim = blockDim;
    this->kernelArgs = args;

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
}

void PtxInterpreter::funcInterpreter(
    std::map<std::string, Symtable *> &name2Sym,
    std::map<std::string, int> &label2pc) {
    // Setup symbols
    setupConstantSymbols(name2Sym);
    setupKernelArguments(name2Sym);
    setupLabels(label2pc);

    int ctaNum = gridDim.x * gridDim.y * gridDim.z;
    CTAContext cta;
    dim3 blockIdx;
    for (int i = 0; i < ctaNum; i++) {
        blockIdx.z = i / (gridDim.x * gridDim.y);
        blockIdx.y = i % (gridDim.x * gridDim.y) / (gridDim.x);
        blockIdx.x = i % (gridDim.x * gridDim.y) % (gridDim.x);
        cta.init(gridDim, blockDim, blockIdx, kernelContext->kernelStatements,
                 name2Sym, label2pc);
        while (cta.exe_once() != EXIT)
            ;
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
    for (int i = 0; i < kernelContext->kernelParams.size(); i++) {
        // temporily ignore align
        auto e = kernelContext->kernelParams[i];
        Symtable *s = new Symtable();
        s->name = e.paramName;
        s->elementNum = e.paramNum;
        s->symType = e.paramType;
        s->byteNum = Q2bytes(e.paramType);
        s->val = (uint64_t)kernelArgs[i];
        name2Sym[s->name] = s;
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