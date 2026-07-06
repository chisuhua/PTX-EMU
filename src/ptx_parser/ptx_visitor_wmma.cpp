// WMMA_INSTR 类别的实现
#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/statement_factory.h"
#include "utils/logger.h"

using namespace ptxir::factory;

#define VISITOR_WMMA_INSTR(openum, opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                                  \
    WmmaType wmmaType = WMMA_MMA;                                                \
    if (ctx->wmmaOp()) {                                                         \
        if (ctx->wmmaOp()->LOAD()) {                                            \
            wmmaType = WMMA_LOAD;                                               \
        } else if (ctx->wmmaOp()->STORE()) {                                    \
            wmmaType = WMMA_STORE;                                              \
        } else if (ctx->wmmaOp()->MMA()) {                                      \
            wmmaType = WMMA_MMA;                                                \
        }                                                                       \
    }                                                                          \
                                                                                  \
    std::vector<OperandContext> operands;                                       \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
    for (size_t i = 0; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) {   \
        operands.push_back(createOperandFromContext(operandCtxs[i]));            \
    }                                                                          \
                                                                                  \
    auto stmtCtx = makeWmmaInstr(wmmaType, extractQualifiersFromContext(ctx),    \
                                  operands, ctx->getText());                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                          \
                                                                                  \
    return nullptr;                                                              \
}

// X-Macro展开

// visitTcgen05Inst: Blackwell sm_100+ tensor core generator (ADR-0016).
// Hand-maintained (not via VISITOR_TCGEN_INSTR X-Macro) because the grammar
// defines a single tcgen05Inst rule, not 11 per-sub-op rules. Full qualifier
// and operand extraction is deferred to change-3 (handler implementation);
// this minimal version prevents the silent-drop that Metis review flagged.
std::any PtxVisitor::visitTcgen05Inst(ptxparser::ptxParser::Tcgen05InstContext *ctx) {
    if (!currentKernel) return nullptr;

    Tcgen05OpKind op_kind = Tcgen05OpKind::MMA;
    if (ctx->tcgen05SubOp()) {
        if (ctx->tcgen05SubOp()->MMA())              op_kind = Tcgen05OpKind::MMA;
        else if (ctx->tcgen05SubOp()->LD())           op_kind = Tcgen05OpKind::LD;
        else if (ctx->tcgen05SubOp()->ST())           op_kind = Tcgen05OpKind::ST;
        else if (ctx->tcgen05SubOp()->TCGEN05_CP())  op_kind = Tcgen05OpKind::CP;
        else if (ctx->tcgen05SubOp()->TCGEN05_ALLOC())    op_kind = Tcgen05OpKind::ALLOC;
        else if (ctx->tcgen05SubOp()->TCGEN05_DEALLOC())  op_kind = Tcgen05OpKind::DEALLOC;
        else if (ctx->tcgen05SubOp()->TCGEN05_RELINQUISH()) op_kind = Tcgen05OpKind::RELINQUISH;
        else if (ctx->tcgen05SubOp()->TCGEN05_COMMIT())  op_kind = Tcgen05OpKind::COMMIT;
        else if (ctx->tcgen05SubOp()->TCGEN05_WAIT())    op_kind = Tcgen05OpKind::WAIT;
        else if (ctx->tcgen05SubOp()->FENCE())        op_kind = Tcgen05OpKind::FENCE;
    }

    std::vector<Qualifier> qualifiers = extractQualifiersFromContext(ctx);
    std::vector<OperandContext> operands;  // TODO(change-3): full operand extraction

    currentKernel->kernelStatements.push_back(
        makeTcgen05Instr(op_kind, qualifiers, operands, ctx->getText()));
    return nullptr;
}
