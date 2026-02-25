// Warp相关指令的实现（VOTE_INSTR, SHFL_INSTR）

#define VISITOR_VOTE_INSTR(openum, opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    VoteInstr vote;                                                            \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    vote.qualifiers = qualifiers;                                              \
    vote.mode = "";                                                            \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        vote.operands.push_back(oc);                                           \
    }                                                                          \
    stmtCtx.data = vote;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_SHFL_INSTR(openum, opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    ShflInstr shfl;                                                            \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    shfl.qualifiers = qualifiers;                                              \
    shfl.mode = "";                                                            \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        shfl.operands.push_back(oc);                                           \
    }                                                                          \
    stmtCtx.data = shfl;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

// X-Macro展开
