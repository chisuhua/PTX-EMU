// ATOM_INSTR 类别的实现
#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/statement_factory.h"
#include "utils/logger.h"

using namespace ptxir::factory;

#define VISITOR_ATOM_INSTR(openum, opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                                    \
    std::vector<OperandContext> operands;                                          \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) {   \
        operands.push_back(createOperandFromContext(operandCtxs[i]));              \
    }                                                                               \
                                                                                    \
    auto stmtCtx = makeAtomInstr(extractQualifiersFromContext(ctx), operands,      \
                                 std::min((int)operandCtxs.size(), (int)opcount),  \
                                 ctx->getText());                                    \
    currentKernel->kernelStatements.push_back(stmtCtx);                            \
                                                                                    \
    return nullptr;                                                                 \
}

// X-Macro展开
