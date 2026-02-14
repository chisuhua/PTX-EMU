// 异步指令的实现（ASYNC_STORE, ASYNC_REDUCE）

#define VISITOR_ASYNC_STORE(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    AsyncStoreInstr asyncStore;                                                \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    asyncStore.qualifiers = qualifiers;                                        \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        asyncStore.operands.push_back(oc);                                     \
    }                                                                          \
    stmtCtx.data = asyncStore;                                                 \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_ASYNC_REDUCE(opstr, opname, opcount)                           \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    AsyncReduceInstr asyncReduce;                                              \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    asyncReduce.qualifiers = qualifiers;                                       \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        asyncReduce.operands.push_back(oc);                                    \
    }                                                                          \
    stmtCtx.data = asyncReduce;                                                \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}
