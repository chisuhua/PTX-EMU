// Tensor相关指令的实现（TCGEN_INSTR, TENSORMAP_INSTR）

#define VISITOR_TCGEN_INSTR(openum, opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    TcgenInstr tcgen;                                                          \
    tcgen.opName = #opstr;                                                     \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    tcgen.qualifiers = qualifiers;                                             \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        tcgen.operands.push_back(oc);                                          \
    }                                                                          \
    stmtCtx.data = tcgen;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_TENSORMAP_INSTR(openum, opstr, opname, opcount)                        \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
    TensormapInstr tensormap;                                                  \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    tensormap.qualifiers = qualifiers;                                         \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        tensormap.operands.push_back(oc);                                      \
    }                                                                          \
    stmtCtx.data = tensormap;                                                  \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

// X-Macro展开
