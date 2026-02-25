// GENERIC_INSTR 类别的实现
#define VISITOR_GENERIC_INSTR(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *c) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                                \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = c->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
                                                                                \
    GenericInstr instr;                                                        \
                                                                                \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(c);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                                \
    /* 提取操作数 */                                                           \
    auto operands = c->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        instr.operands.push_back(oc);                                          \
    }                                                                          \
                                                                                \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                                \
    return nullptr;                                                            \
}

// X-Macro passes 6 params, we need to map to our 3-param macro
#define X(a, b, c, d, e, f) VISITOR_GENERIC_INSTR(c, b, d)
#include "ptx_ir/ptx_op.def"
#undef X
