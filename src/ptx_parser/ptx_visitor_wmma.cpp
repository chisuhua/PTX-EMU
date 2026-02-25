// WMMA_INSTR 类别的实现
#define VISITOR_WMMA_INSTR(openum, opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                                 \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
                                                                                 \
    WmmaInstr instr;                                                           \
                                                                                 \
    /* 确定WMMA类型 */                                                         \
    if (ctx->wmmaOp()) {                                                         \
        if (ctx->wmmaOp()->LOAD()) {                                            \
            instr.wmmaType = WMMA_LOAD;                                         \
        } else if (ctx->wmmaOp()->STORE()) {                                    \
            instr.wmmaType = WMMA_STORE;                                        \
        } else if (ctx->wmmaOp()->MMA()) {                                     \
            instr.wmmaType = WMMA_MMA;                                          \
        }                                                                       \
    }                                                                          \
                                                                                 \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                                 \
    /* 提取操作数 */                                                           \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        instr.operands.push_back(oc);                                          \
    }                                                                          \
                                                                                 \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                                 \
    return nullptr;                                                            \
}

// X-Macro展开
