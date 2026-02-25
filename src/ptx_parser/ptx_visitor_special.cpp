// 特殊指令类别的实现（PREDICATE_PREFIX, MEMBAR_INSTR, FENCE_INSTR, REDUX_INSTR, MBARRIER_INSTR）

#define VISITOR_PREDICATE_PREFIX(openum, opstr, opname, opcount)                       \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    PredicatePrefix pred;                                                      \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    pred.qualifiers = qualifiers;                                              \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        pred.operands.push_back(oc);                                           \
    }                                                                          \
    pred.target = "";                                                          \
    stmtCtx.data = pred;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_MEMBAR_INSTR(openum, opstr, opname, opcount)                           \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    MembarInstr membar;                                                        \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    membar.qualifiers = qualifiers;                                            \
    membar.scope = "";                                                         \
    stmtCtx.data = membar;                                                     \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_FENCE_INSTR(openum, opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    FenceInstr fence;                                                          \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    fence.qualifiers = qualifiers;                                              \
    fence.memoryOrder = "";                                                    \
    fence.scope = "";                                                          \
    stmtCtx.data = fence;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_REDUX_INSTR(openum, opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    ReduxSyncInstr redux;                                                      \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    redux.qualifiers = qualifiers;                                              \
    redux.operation = "";                                                      \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        redux.operands.push_back(oc);                                          \
    }                                                                          \
    stmtCtx.data = redux;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_MBARRIER_INSTR(openum, opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    MbarrierInstr mbarrier;                                                    \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    mbarrier.qualifiers = qualifiers;                                          \
    mbarrier.operation = "";                                                   \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        mbarrier.operands.push_back(oc);                                       \
    }                                                                          \
    stmtCtx.data = mbarrier;                                                  \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

// X-Macro展开
