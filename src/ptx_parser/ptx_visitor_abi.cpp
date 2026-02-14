// ABI指令的实现（ABI_DIRECTIVE）

#define VISITOR_ABI_DIRECTIVE(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    AbiDirective abiDir;                                                       \
    abiDir.regNumber = 0;                                                      \
    stmtCtx.data = abiDir;                                                     \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}
