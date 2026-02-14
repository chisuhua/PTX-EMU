// 特殊指令类别的实现（PREDICATE_PREFIX, MEMBAR_INSTR, FENCE_INSTR, REDUX_INSTR, MBARRIER_INSTR）

#define VISITOR_PREDICATE_PREFIX(opstr, opname, opcount)                       \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    PredicatePrefix pred;                                                      \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    pred.qualifiers = qualifiers;                                              \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        pred.operands.push_back(oc);                                           \
    }                                                                          \
    pred.target = "";                                                          \
    stmtCtx.data = pred;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_MEMBAR_INSTR(opstr, opname, opcount)                           \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    MembarInstr membar;                                                        \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    membar.qualifiers = qualifiers;                                            \
    membar.scope = "";                                                         \
    stmtCtx.data = membar;                                                     \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_FENCE_INSTR(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    FenceInstr fence;                                                          \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    fence.qualifiers = qualifiers;                                             \
    fence.memoryOrder = "";                                                    \
    fence.scope = "";                                                          \
    stmtCtx.data = fence;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_REDUX_INSTR(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    ReduxSyncInstr redux;                                                      \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    redux.qualifiers = qualifiers;                                             \
    redux.operation = "";                                                      \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        redux.operands.push_back(oc);                                          \
    }                                                                          \
    stmtCtx.data = redux;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_MBARRIER_INSTR(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    MbarrierInstr mbarrier;                                                    \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    mbarrier.qualifiers = qualifiers;                                          \
    mbarrier.operation = "";                                                   \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        mbarrier.operands.push_back(oc);                                       \
    }                                                                          \
    stmtCtx.data = mbarrier;                                                   \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}
