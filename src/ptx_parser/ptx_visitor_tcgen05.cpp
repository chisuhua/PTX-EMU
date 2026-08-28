// 包含 Blackwell tcgen05 visitor 实现 (ADR-0016)
#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/statement_factory.h"

using namespace ptxir::factory;

std::any PtxVisitor::visitTcgen05Inst(ptxparser::ptxParser::Tcgen05InstContext *ctx) {
    if (!currentKernel) return nullptr;

    ptxemu::ir::Tcgen05OpKind op_kind = ptxemu::ir::Tcgen05OpKind::MMA;
    if (ctx->tcgen05SubOp()) {
        if (ctx->tcgen05SubOp()->MMA())              op_kind = ptxemu::ir::Tcgen05OpKind::MMA;
        else if (ctx->tcgen05SubOp()->LD())           op_kind = ptxemu::ir::Tcgen05OpKind::LD;
        else if (ctx->tcgen05SubOp()->ST())           op_kind = ptxemu::ir::Tcgen05OpKind::ST;
        else if (ctx->tcgen05SubOp()->TCGEN05_CP())  op_kind = ptxemu::ir::Tcgen05OpKind::CP;
        else if (ctx->tcgen05SubOp()->TCGEN05_ALLOC())    op_kind = ptxemu::ir::Tcgen05OpKind::ALLOC;
        else if (ctx->tcgen05SubOp()->TCGEN05_DEALLOC())  op_kind = ptxemu::ir::Tcgen05OpKind::DEALLOC;
        else if (ctx->tcgen05SubOp()->TCGEN05_RELINQUISH()) op_kind = ptxemu::ir::Tcgen05OpKind::RELINQUISH;
        else if (ctx->tcgen05SubOp()->TCGEN05_COMMIT())  op_kind = ptxemu::ir::Tcgen05OpKind::COMMIT;
        else if (ctx->tcgen05SubOp()->TCGEN05_WAIT())    op_kind = ptxemu::ir::Tcgen05OpKind::WAIT;
        else if (ctx->tcgen05SubOp()->FENCE())        op_kind = ptxemu::ir::Tcgen05OpKind::FENCE;
    }

    std::vector<ptxemu::ir::Qualifier> qualifiers = extractQualifiersFromContext(ctx);

    // C3 fix: extract cta_group IMMEDIATE value from parse tree.
    // Grammar: TCGEN_CTA_GROUP COLONCOLON IMMEDIATE (ptxInstructions.g4:451).
    // extractQualifiersFromContext drops the IMMEDIATE child silently
    // (tokenToQualifier returns Q_UNKNOWN for terminal nodes).
    // Per Oracle Q5 Option (b): add separate parse-tree walk here instead
    // of modifying extractQualifiersFromContext (which has 19 call sites).
    uint32_t cta_group = 1;
    if (ctx->tcgen05Qual().size()) {
        for (auto* qualCtx : ctx->tcgen05Qual()) {
            if (qualCtx->TCGEN_CTA_GROUP() && qualCtx->IMMEDIATE()) {
                cta_group = static_cast<uint32_t>(
                    std::stoul(qualCtx->IMMEDIATE()->getText()));
            }
        }
    }

    std::vector<ptxemu::ir::OperandContext> operands;
    auto opListCtx = ctx->tcgen05Operands();
    if (opListCtx) {
        for (auto* opCtx : opListCtx->tcgen05Operand()) {
            if (!opCtx) continue;
            if (opCtx->vectorRegister()) {
                auto* vr = opCtx->vectorRegister();
                std::string text = vr->getText();
                if (!text.empty() && text.front() == '{') text.erase(0, 1);
                if (!text.empty() && text.back() == '}') text.pop_back();
                operands.push_back(ptxemu::ir::OperandContext(VariableOperand{text}));
            } else if (opCtx->address()) {
                operands.push_back(
                    std::any_cast<ptxemu::ir::OperandContext>(
                        visitAddress(opCtx->address())));
            } else if (opCtx->operand()) {
                operands.push_back(
                    createOperandFromContext(opCtx->operand()));
            }
        }
    }

    currentKernel->kernelStatements.push_back(
        makeTcgen05Instr(op_kind, qualifiers, operands, ctx->getText(),
                         cta_group));
    return nullptr;
}