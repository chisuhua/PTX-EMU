// CALL_INSTR 类别的实现
#include <any>

#define VISITOR_CALL_INSTR(openum, opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                                   \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
                                                                                   \
    CallInstr instr;                                                           \
                                                                                   \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                                   \
    /* 提取函数名 (从labelOperand) */                                          \
    if (ctx->labelOperand()) {                                                 \
        instr.funcName = ctx->labelOperand()->getText();                       \
    }                                                                           \
                                                                                   \
    /* 提取操作数 */                                                           \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        instr.operands.push_back(oc);                                          \
    }                                                                                  \
                                                                                   \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                                   \
    return nullptr;                                                            \
}

// call.uni instruction visitor
std::any PtxVisitor::visitCallUniInst(ptxparser::ptxParser::CallUniInstContext *ctx) {
    if (!currentKernel) return nullptr;

    StatementContext stmtCtx;
    stmtCtx.instructionText = ctx->getText();
    stmtCtx.type = S_CALL;

    CallInstr instr;

    // Extract qualifiers
    auto qualifiers = extractQualifiersFromContext(ctx);
    instr.qualifiers = qualifiers;

    // Function name is in labelOperand
    if (ctx->labelOperand()) {
        instr.funcName = ctx->labelOperand()->getText();
    }

    // Store instruction text for runtime access
    instr.instructionText = ctx->getText();

    // Extract operands (the call arguments)
    for (size_t i = 0; i < ctx->operand().size(); i++) {
        auto oc = createOperandFromContext(ctx->operand(i));
        instr.operands.push_back(oc);
    }

    stmtCtx.data = instr;
    currentKernel->kernelStatements.push_back(stmtCtx);

    return nullptr;
}

// X-Macro展开
