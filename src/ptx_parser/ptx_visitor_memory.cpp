// 内存相关指令的实现（TEXTURE_INSTR, SURFACE_INSTR, REDUCTION_INSTR, PREFETCH_INSTR, CP_ASYNC_INSTR）

#define VISITOR_TEXTURE_INSTR(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    TextureInstr tex;                                                          \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    tex.qualifiers = qualifiers;                                               \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        tex.operands.push_back(oc);                                            \
    }                                                                          \
    stmtCtx.data = tex;                                                        \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_SURFACE_INSTR(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    SurfaceInstr surf;                                                         \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    surf.qualifiers = qualifiers;                                              \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        surf.operands.push_back(oc);                                           \
    }                                                                          \
    stmtCtx.data = surf;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_REDUCTION_INSTR(opstr, opname, opcount)                        \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    ReductionInstr red;                                                        \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    red.qualifiers = qualifiers;                                               \
    red.operation = "";                                                        \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        red.operands.push_back(oc);                                            \
    }                                                                          \
    stmtCtx.data = red;                                                        \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_PREFETCH_INSTR(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    PrefetchInstr prefetch;                                                    \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    prefetch.qualifiers = qualifiers;                                          \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        prefetch.operands.push_back(oc);                                       \
    }                                                                          \
    stmtCtx.data = prefetch;                                                   \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_CP_ASYNC_INSTR(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    CpAsyncInstr cpAsync;                                                      \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    cpAsync.qualifiers = qualifiers;                                           \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        cpAsync.operands.push_back(oc);                                        \
    }                                                                          \
    stmtCtx.data = cpAsync;                                                    \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}
