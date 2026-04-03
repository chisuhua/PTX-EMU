// BARRIER 类别的实现

// VISITOR_BARRIER macro for X-macro in ptx_visitor.cpp (handles regular bar.sync, bar.arrive, etc.)
#define VISITOR_BARRIER(openum, opstr, opname, opcount) \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) { \
    if (!currentKernel) return nullptr; \
    \
    StatementContext stmtCtx; \
    stmtCtx.instructionText = ctx->getText(); \
    stmtCtx.type = openum; \
    \
    BarrierInstr instr; \
    instr.qualifiers = extractQualifiersFromContext(ctx); \
    \
    /* Extract bar ID from barrierOperands if present */ \
    if (ctx->barrierOperands() && ctx->barrierOperands()->IMMEDIATE()) { \
        instr.barId = extractIntFromToken(ctx->barrierOperands()->IMMEDIATE()->getSymbol()); \
    } \
    \
    stmtCtx.data = instr; \
    currentKernel->kernelStatements.push_back(stmtCtx); \
    \
    return nullptr; \
}

// VISITOR_WARP_BARRIER: Manual implementation of bar.warp.sync instruction
std::any PtxVisitor::visitBarWarpSyncInst(ptxparser::ptxParser::BarWarpSyncInstContext *ctx) {
    if (!currentKernel) return nullptr;
    
    StatementContext stmtCtx;
    stmtCtx.instructionText = ctx->getText();
    stmtCtx.type = S_BAR_WARP_SYNC;
    
    BarWarpSyncInstr instr;
    instr.qualifiers = extractQualifiersFromContext(ctx);
    
    // Parse participation mask operand
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();
    if (!operands.empty()) {
        instr.operands.push_back(createOperandFromContext(operands[0]));
    }
    
    // Parse reconvergence label
    if (ctx->labelOperand()) {
        instr.reconvergenceLabel = ctx->labelOperand()->ID()->getText();
    }
    
    stmtCtx.data = instr;
    currentKernel->kernelStatements.push_back(stmtCtx);
    
    return nullptr;
}
