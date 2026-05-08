// 内存相关指令的实现（TEXTURE_INSTR, SURFACE_INSTR, REDUCTION_INSTR, PREFETCH_INSTR, CP_ASYNC_INSTR）
#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/statement_factory.h"
#include "utils/logger.h"

using namespace ptxir::factory;

#define VISITOR_TEXTURE_INSTR(openum, opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    std::vector<OperandContext> operands;                                       \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) {   \
        operands.push_back(createOperandFromContext(operandCtxs[i]));            \
    }                                                                          \
    auto stmtCtx = makeTextureInstr(extractQualifiersFromContext(ctx), operands, ctx->getText()); \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_SURFACE_INSTR(openum, opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    std::vector<OperandContext> operands;                                       \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) {   \
        operands.push_back(createOperandFromContext(operandCtxs[i]));            \
    }                                                                          \
    auto stmtCtx = makeSurfaceInstr(extractQualifiersFromContext(ctx), operands, ctx->getText()); \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_REDUCTION_INSTR(openum, opstr, opname, opcount)                        \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    std::vector<OperandContext> operands;                                       \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) {   \
        operands.push_back(createOperandFromContext(operandCtxs[i]));            \
    }                                                                          \
    auto stmtCtx = makeReductionInstr(extractQualifiersFromContext(ctx), "", operands, ctx->getText()); \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_PREFETCH_INSTR(openum, opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    std::vector<OperandContext> operands;                                       \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) {   \
        operands.push_back(createOperandFromContext(operandCtxs[i]));            \
    }                                                                          \
    auto stmtCtx = makePrefetchInstr(extractQualifiersFromContext(ctx), operands, ctx->getText()); \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_CP_ASYNC_INSTR(openum, opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    std::vector<OperandContext> operands;                                       \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) {   \
        operands.push_back(createOperandFromContext(operandCtxs[i]));            \
    }                                                                          \
    auto stmtCtx = makeCpAsyncInstr(extractQualifiersFromContext(ctx), operands, ctx->getText()); \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

// X-Macro展开
