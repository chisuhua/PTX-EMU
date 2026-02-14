// BRANCH 类别的实现
#define VISITOR_BRANCH(opstr, opname, opcount)                                 \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                               \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
                                                                               \
    BranchInstr instr;                                                         \
                                                                               \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                               \
    /* 提取跳转目标 */                                                         \
    if (ctx->ID()) {                                                           \
        instr.target = ctx->ID()->getText();                                   \
    }                                                                          \
                                                                               \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                               \
    return nullptr;                                                            \
}
