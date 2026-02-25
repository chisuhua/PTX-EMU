// 内存相关指令的实现（TEXTURE_INSTR, SURFACE_INSTR, REDUCTION_INSTR, PREFETCH_INSTR, CP_ASYNC_INSTR）

#define VISITOR_TEXTURE_INSTR(openum, opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    TextureInstr tex;                                                          \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    tex.qualifiers = qualifiers;                                               \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        tex.operands.push_back(oc);                                            \
    }                                                                          \
    stmtCtx.data = tex;                                                        \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_SURFACE_INSTR(openum, opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    SurfaceInstr surf;                                                         \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    surf.qualifiers = qualifiers;                                              \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        surf.operands.push_back(oc);                                           \
    }                                                                          \
    stmtCtx.data = surf;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_REDUCTION_INSTR(openum, opstr, opname, opcount)                        \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    ReductionInstr red;                                                        \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    red.qualifiers = qualifiers;                                               \
    red.operation = "";                                                        \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        red.operands.push_back(oc);                                            \
    }                                                                          \
    stmtCtx.data = red;                                                        \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_PREFETCH_INSTR(openum, opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    PrefetchInstr prefetch;                                                    \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    prefetch.qualifiers = qualifiers;                                          \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        prefetch.operands.push_back(oc);                                       \
    }                                                                          \
    stmtCtx.data = prefetch;                                                   \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_CP_ASYNC_INSTR(openum, opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    CpAsyncInstr cpAsync;                                                      \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    cpAsync.qualifiers = qualifiers;                                           \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        cpAsync.operands.push_back(oc);                                        \
    }                                                                          \
    stmtCtx.data = cpAsync;                                                    \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

// X-Macro展开
