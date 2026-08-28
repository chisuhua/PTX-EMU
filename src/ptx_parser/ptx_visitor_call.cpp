// CALL_INSTR 类别的实现
#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/statement_factory.h"
#include "utils/logger.h"

using namespace ptxir::factory;

#define VISITOR_CALL_INSTR(openum, opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                                    \
    std::string funcName;                                                          \
    if (ctx->labelOperand()) {                                                     \
        funcName = ctx->labelOperand()->getText();                                \
    }                                                                               \
                                                                                    \
    std::vector<ptxemu::ir::OperandContext> operands;                                          \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) {   \
        operands.push_back(createOperandFromContext(operandCtxs[i]));              \
    }                                                                               \
                                                                                    \
    auto stmtCtx = makeCallInstr(openum, funcName, extractQualifiersFromContext(ctx), \
                                 operands, ctx->getText());                         \
    currentKernel->kernelStatements.push_back(stmtCtx);                            \
                                                                                    \
    return nullptr;                                                                 \
}

// call.uni instruction visitor
std::any PtxVisitor::visitCallUniInst(ptxparser::ptxParser::CallUniInstContext *ctx) {
    if (!currentKernel) return nullptr;

    std::string funcName;
    if (ctx->labelOperand()) {
        funcName = ctx->labelOperand()->getText();
    }

    std::vector<ptxemu::ir::OperandContext> operands;
    for (size_t i = 0; i < ctx->operand().size(); i++) {
        operands.push_back(createOperandFromContext(ctx->operand(i)));
    }

    auto stmtCtx = makeCallInstr(S_CALL, funcName, extractQualifiersFromContext(ctx),
                                 operands, ctx->getText());
    stmtCtx.get<CallInstr>().instructionText = ctx->getText();
    currentKernel->kernelStatements.push_back(stmtCtx);

    return nullptr;
}

// X-Macro展开
