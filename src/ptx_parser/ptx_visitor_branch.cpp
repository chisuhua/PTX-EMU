// BRANCH 类别的实现
#define VISITOR_BRANCH(openum, opstr, opname, opcount)                                 \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                                 \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
                                                                                 \
    BranchInstr instr;                                                         \
                                                                                 \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                                 \
    /* 提取跳转目标 */                                                         \
    if (ctx->labelOperand()) {                                                           \
        instr.target = ctx->labelOperand()->getText();                                   \
    }                                                                          \
                                                                                 \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                                 \
    return nullptr;                                                            \
}

// X-Macro展开
