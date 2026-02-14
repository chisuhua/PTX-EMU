// Tensor相关指令的实现（TCGEN_INSTR, TENSORMAP_INSTR）

#define VISITOR_TCGEN_INSTR(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    TcgenInstr tcgen;                                                          \
    tcgen.opName = #opstr;                                                     \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    tcgen.qualifiers = qualifiers;                                             \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        tcgen.operands.push_back(oc);                                          \
    }                                                                          \
    stmtCtx.data = tcgen;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_TENSORMAP_INSTR(opstr, opname, opcount)                        \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    TensormapInstr tensormap;                                                  \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    tensormap.qualifiers = qualifiers;                                         \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        tensormap.operands.push_back(oc);                                      \
    }                                                                          \
    stmtCtx.data = tensormap;                                                  \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}
