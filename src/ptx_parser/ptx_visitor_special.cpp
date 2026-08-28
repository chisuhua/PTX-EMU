// 特殊指令类别的实现（PREDICATE_PREFIX, MEMBAR_INSTR, FENCE_INSTR, REDUX_INSTR, MBARRIER_INSTR）
#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/statement_factory.h"
#include "utils/logger.h"

using namespace ptxir::factory;

#define VISITOR_PREDICATE_PREFIX(openum, opstr, opname, opcount)                       \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    std::vector<ptxemu::ir::OperandContext> operands;                                       \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) {   \
        operands.push_back(createOperandFromContext(operandCtxs[i]));            \
    }                                                                          \
    auto stmtCtx = makePredicatePrefix(extractQualifiersFromContext(ctx), operands, "", ctx->getText()); \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_MEMBAR_INSTR(openum, opstr, opname, opcount)                           \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    auto stmtCtx = makeMembarInstr(extractQualifiersFromContext(ctx), "", ctx->getText()); \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_FENCE_INSTR(openum, opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    auto stmtCtx = makeFenceInstr(extractQualifiersFromContext(ctx), "", "", ctx->getText()); \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_REDUX_INSTR(openum, opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    std::vector<ptxemu::ir::OperandContext> operands;                                       \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) {   \
        operands.push_back(createOperandFromContext(operandCtxs[i]));            \
    }                                                                          \
    auto stmtCtx = makeReduxSyncInstr(extractQualifiersFromContext(ctx), "", operands, ctx->getText()); \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_MBARRIER_INSTR(openum, opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    std::vector<ptxemu::ir::OperandContext> operands;                                       \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) {   \
        operands.push_back(createOperandFromContext(operandCtxs[i]));            \
    }                                                                          \
    auto stmtCtx = makeMbarrierInstr(openum, extractQualifiersFromContext(ctx), "", operands, ctx->getText()); \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

// X-Macro展开
