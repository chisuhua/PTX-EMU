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
        std::string labelName = ctx->labelOperand()->getText();                          \
        if (!labelName.empty() && labelName[0] == '$') {                                 \
            labelName = labelName.substr(1);                                             \
        }                                                                                \
        instr.target = labelName;                                                        \
    }                                                                          \
                                                                                 \
    /* 提取谓词条件 (可选) */                                                 \
    if (ctx->predicate()) {                                                           \
        if (ctx->predicate()->BANG()) {                                              \
            instr.predicate_negated = true;                                          \
        }                                                                              \
        if (ctx->predicate()->operand()) {                                           \
            std::string predName = ctx->predicate()->operand()->getText();            \
            instr.predicate = predName;                                              \
        }                                                                              \
    }                                                                          \
                                                                                 \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                                 \
    return nullptr;                                                            \
}

// X-Macro展开
