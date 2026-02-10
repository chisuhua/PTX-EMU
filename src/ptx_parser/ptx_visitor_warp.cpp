// Warp相关指令的实现（VOTE_INSTR, SHFL_INSTR）

#define VISITOR_VOTE_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    VoteInstr vote;                                                            \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    vote.qualifiers = qualifiers;                                              \
    vote.mode = "";                                                            \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        vote.operands.push_back(oc);                                           \
    }                                                                          \
    stmtCtx.data = vote;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_SHFL_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    ShflInstr shfl;                                                            \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    shfl.qualifiers = qualifiers;                                              \
    shfl.mode = "";                                                            \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        shfl.operands.push_back(oc);                                           \
    }                                                                          \
    stmtCtx.data = shfl;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}
