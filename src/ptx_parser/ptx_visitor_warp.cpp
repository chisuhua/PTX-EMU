// Warp 相关指令的实现（VOTE_INSTR, SHFL_INSTR）
#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/statement_factory.h"
#include "utils/logger.h"

using namespace ptxir::factory;

// VOTE instruction visitor
#define VISITOR_VOTE_INSTR(openum, opstr, opname, opcount) \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) { \
    if (!currentKernel) return nullptr; \
    \
    std::string mode; \
    if (ctx->voteMode()) { \
        mode = ctx->voteMode()->getText(); \
    } \
    \
    std::vector<OperandContext> operands; \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>(); \
    for (const auto& operandCtx : operandCtxs) { \
        operands.push_back(createOperandFromContext(operandCtx)); \
    } \
    \
    auto stmtCtx = makeVoteInstr(extractQualifiersFromContext(ctx), mode, operands, ctx->getText()); \
    currentKernel->kernelStatements.push_back(stmtCtx); \
    \
    return nullptr; \
}

// SHFL instruction visitor
#define VISITOR_SHFL_INSTR(openum, opstr, opname, opcount) \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) { \
    if (!currentKernel) return nullptr; \
    \
    std::string mode; \
    if (ctx->shuffleMode()) { \
        mode = ctx->shuffleMode()->getText(); \
    } \
    \
    std::vector<OperandContext> operands; \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>(); \
    for (const auto& operandCtx : operandCtxs) { \
        operands.push_back(createOperandFromContext(operandCtx)); \
    } \
    \
    auto stmtCtx = makeShflInstr(extractQualifiersFromContext(ctx), mode, operands, ctx->getText()); \
    currentKernel->kernelStatements.push_back(stmtCtx); \
    \
    return nullptr; \
}

// WARP_BARRIER: bar.warp.sync - manually implemented, macro is just a placeholder
#define VISITOR_WARP_BARRIER(openum, opstr, opname, opcount)
