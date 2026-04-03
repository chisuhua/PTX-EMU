// Warp 相关指令的实现（VOTE_INSTR, SHFL_INSTR, WARP_BARRIER, GENERIC_INSTR）

// VOTE instruction visitor
#define VISITOR_VOTE_INSTR(openum, opstr, opname, opcount) \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) { \
    if (!currentKernel) return nullptr; \
    \
    StatementContext stmtCtx; \
    stmtCtx.instructionText = ctx->getText(); \
    stmtCtx.type = openum; \
    \
    VoteInstr vote; \
    vote.qualifiers = extractQualifiersFromContext(ctx); \
    \
    if (ctx->voteMode()) { \
        vote.mode = ctx->voteMode()->getText(); \
    } \
    \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>(); \
    for (const auto& operandCtx : operands) { \
        vote.operands.push_back(createOperandFromContext(operandCtx)); \
    } \
    \
    stmtCtx.data = vote; \
    currentKernel->kernelStatements.push_back(stmtCtx); \
    \
    return nullptr; \
}

// SHFL instruction visitor
#define VISITOR_SHFL_INSTR(openum, opstr, opname, opcount) \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) { \
    if (!currentKernel) return nullptr; \
    \
    StatementContext stmtCtx; \
    stmtCtx.instructionText = ctx->getText(); \
    stmtCtx.type = openum; \
    \
    ShflInstr shfl; \
    shfl.qualifiers = extractQualifiersFromContext(ctx); \
    \
    if (ctx->shuffleMode()) { \
        shfl.mode = ctx->shuffleMode()->getText(); \
    } \
    \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>(); \
    for (const auto& operandCtx : operands) { \
        shfl.operands.push_back(createOperandFromContext(operandCtx)); \
    } \
    \
    stmtCtx.data = shfl; \
    currentKernel->kernelStatements.push_back(stmtCtx); \
    \
    return nullptr; \
}

// WARP_BARRIER: bar.warp.sync - manually implemented, macro is just a placeholder
// Activemask uses GENERIC_INSTR which is handled by ptx_visitor_generic.cpp
#define VISITOR_WARP_BARRIER(openum, opstr, opname, opcount) \
/* BarWarpSync manually implemented in ptx_visitor_barrier.cpp */
