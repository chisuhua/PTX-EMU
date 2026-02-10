// 简单指令类别的实现（OPERAND_REG, OPERAND_CONST, OPERAND_MEMORY, SIMPLE_NAME, SIMPLE_STRING, VOID_INSTR）

#define VISITOR_OPERAND_REG(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    DeclarationInstr decl;                                                     \
    decl.kind = DeclarationInstr::Kind::REG;                                   \
    decl.name = "";                                                            \
    stmtCtx.data = decl;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_OPERAND_CONST(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    DeclarationInstr decl;                                                     \
    decl.kind = DeclarationInstr::Kind::CONST;                                 \
    decl.name = "";                                                            \
    stmtCtx.data = decl;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_OPERAND_MEMORY(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    DeclarationInstr decl;                                                     \
    if (opstr == "SHARED") decl.kind = DeclarationInstr::Kind::SHARED;         \
    else if (opstr == "LOCAL") decl.kind = DeclarationInstr::Kind::LOCAL;      \
    else if (opstr == "GLOBAL") decl.kind = DeclarationInstr::Kind::GLOBAL;    \
    else if (opstr == "PARAM") decl.kind = DeclarationInstr::Kind::PARAM;      \
    else decl.kind = DeclarationInstr::Kind::GLOBAL;                           \
    decl.name = "";                                                            \
    stmtCtx.data = decl;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_SIMPLE_NAME(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    DollarNameInstr dollar;                                                    \
    dollar.name = "";                                                          \
    stmtCtx.data = dollar;                                                     \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_SIMPLE_STRING(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    PragmaInstr pragma;                                                        \
    pragma.content = "";                                                       \
    stmtCtx.data = pragma;                                                     \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_VOID_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    VoidInstr voidInstr;                                                       \
    stmtCtx.data = voidInstr;                                                  \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}
