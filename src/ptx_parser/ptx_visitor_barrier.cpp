// BARRIER 类别的实现
#define VISITOR_BARRIER(opstr, opname, opcount)                                \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                               \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
                                                                               \
    BarrierInstr instr;                                                        \
                                                                               \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                               \
    /* 提取barrier ID */                                                       \
    if (ctx->IMMEDIATE()) {                                                    \
        instr.barId = extractIntFromToken(ctx->IMMEDIATE()->getSymbol());      \
    }                                                                          \
                                                                               \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                               \
    return nullptr;                                                            \
}
