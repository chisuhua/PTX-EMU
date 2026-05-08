// 简单指令类别的实现（VOID_INSTR）
#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/statement_factory.h"
#include "utils/logger.h"

using namespace ptxir::factory;

#define VISITOR_VOID_INSTR(openum, opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    auto stmtCtx = makeVoidInstr(openum, ctx->getText());                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_OPERAND_REG(openum, opstr, opname, opcount)
#define VISITOR_OPERAND_CONST(openum, opstr, opname, opcount)
#define VISITOR_OPERAND_MEMORY(openum, opstr, opname, opcount)
#define VISITOR_SIMPLE_NAME(openum, opstr, opname, opcount)
#define VISITOR_SIMPLE_STRING(openum, opstr, opname, opcount)
#define VISITOR_LABEL_INSTR(openum, opstr, opname, opcount)
