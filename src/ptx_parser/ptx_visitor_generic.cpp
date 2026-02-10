// GENERIC_INSTR 类别的实现
#define VISITOR_GENERIC_INSTR(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                               \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
                                                                               \
    GenericInstr instr;                                                        \
                                                                               \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                               \
    /* 提取操作数 */                                                           \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        instr.operands.push_back(oc);                                          \
    }                                                                          \
                                                                               \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                               \
    return nullptr;                                                            \
}
