// WMMA_INSTR 类别的实现
#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/statement_factory.h"
#include "utils/logger.h"

using namespace ptxir::factory;

#define VISITOR_WMMA_INSTR(openum, opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                                  \
    WmmaType wmmaType = WMMA_MMA;                                                \
    if (ctx->wmmaOp()) {                                                         \
        if (ctx->wmmaOp()->LOAD()) {                                            \
            wmmaType = WMMA_LOAD;                                               \
        } else if (ctx->wmmaOp()->STORE()) {                                    \
            wmmaType = WMMA_STORE;                                              \
        } else if (ctx->wmmaOp()->MMA()) {                                      \
            wmmaType = WMMA_MMA;                                                \
        }                                                                       \
    }                                                                          \
                                                                                  \
    std::vector<OperandContext> operands;                                       \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) {   \
        operands.push_back(createOperandFromContext(operandCtxs[i]));            \
    }                                                                          \
                                                                                  \
    auto stmtCtx = makeWmmaInstr(wmmaType, extractQualifiersFromContext(ctx),    \
                                  operands, ctx->getText());                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                          \
                                                                                  \
    return nullptr;                                                              \
}

// X-Macro展开
