// 异步指令的实现（ASYNC_STORE, ASYNC_REDUCE）

#define VISITOR_ASYNC_STORE(openum, opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    AsyncStoreInstr asyncStore;                                                \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    asyncStore.qualifiers = qualifiers;                                        \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        asyncStore.operands.push_back(oc);                                     \
    }                                                                          \
    stmtCtx.data = asyncStore;                                                 \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_ASYNC_REDUCE(openum, opstr, opname, opcount)                           \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    AsyncReduceInstr asyncReduce;                                              \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    asyncReduce.qualifiers = qualifiers;                                        \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        asyncReduce.operands.push_back(oc);                                    \
    }                                                                          \
    stmtCtx.data = asyncReduce;                                               \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

// X-Macro展开
