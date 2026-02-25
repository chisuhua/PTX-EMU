// BARRIER 类别的实现
#define VISITOR_BARRIER(openum, opstr, opname, opcount)                                \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                                 \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
                                                                                 \
    BarrierInstr instr;                                                        \
                                                                                 \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                                 \
    /* 提取barrier ID */                                                       \
    if (ctx->DECIMAL_INT()) {                                                    \
        instr.barId = extractIntFromToken(ctx->DECIMAL_INT()->getSymbol());      \
    }                                                                          \
                                                                                 \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                                 \
    return nullptr;                                                            \
}

// X-Macro展开
