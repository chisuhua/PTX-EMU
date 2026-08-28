// ATOM_INSTR 类别的实现
#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/statement_factory.h"
#include "utils/logger.h"

using namespace ptxir::factory;

#define VISITOR_ATOM_INSTR(openum, opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                                     \
    std::vector<ptxemu::ir::OperandContext> operands;                                          \
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();     \
                                                                                     \
    /* atom grammar:                                                                  \
     *   atomInst: ATOM atomQualifiers atomOp typeSpecifier vectorSpec?              \
     *              operand COMMA addressExpr COMMA operand (COMMA operand)? SEMI     \
     *                                                                                \
     * operandCtxs layout:                                                            \
     *   operandCtxs[0] = dst                                                         \
     *   operandCtxs[1] = src                                                         \
     *   operandCtxs[2] = cmp (optional, only for atom.cas)                           \
     * ctx->addressExpr() = middle address expression (the [addr] part)              \
     *                                                                                \
     * The previous implementation only collected dst+src via                         \
     * getRuleContexts<ptxemu::ir::OperandContext>() and silently dropped the middle             \
     * addressExpr, yielding fewer operands than S_ATOM in ptx_op.def declares.      \
     *                                                                                \
     * Fix: explicitly convert ctx->addressExpr() into an AddrOperand and            \
     * insert it between dst and src so the resulting operands vector                \
     * contains exactly {dst, addr, src[, cmp]}.                                      \
     */                                                                               \
    if (operandCtxs.size() >= 2 && ctx->addressExpr() != nullptr) {              \
        operands.push_back(createOperandFromContext(operandCtxs[0]));              \
        auto *addrExprCtx = ctx->addressExpr();                                    \
        AddrOperand addr;                                                          \
        addr.space = AddrOperand::Space::GLOBAL;                                   \
        addr.offsetType = AddrOperand::OffsetType::IMMEDIATE;                      \
        addr.immediateOffset = "0";                                                \
        if (addrExprCtx->operand()) {                                              \
            auto baseOp = createOperandFromContext(addrExprCtx->operand());        \
            if (baseOp.kind() == ptxemu::ir::OperandKind::REG) {                               \
                const auto &reg = std::get<RegOperand>(baseOp.data);               \
                addr.baseSymbol = reg.fullName();                                  \
                addr.id = reg.fullName();                                          \
                addr.offsetType = AddrOperand::OffsetType::REGISTER;               \
                addr.registerOffset = std::make_shared<ptxemu::ir::OperandContext>(baseOp);    \
            } else {                                                               \
                std::string raw = addrExprCtx->getText();                          \
                if (!raw.empty() && (raw.front() == '[' || raw.back() == ']')) {   \
                    if (raw.front() == '[') raw.erase(raw.begin());                \
                    if (!raw.empty() && raw.back() == ']') raw.pop_back();          \
                }                                                                  \
                if (!raw.empty() && (raw.front() == '%' || raw.front() == '$')) {  \
                    raw.erase(raw.begin());                                        \
                }                                                                  \
                addr.baseSymbol = raw;                                             \
                addr.id = raw;                                                     \
            }                                                                      \
        }                                                                          \
        if (addrExprCtx->immediate()) {                                            \
            auto *immCtx = addrExprCtx->immediate();                               \
            addr.offsetType = AddrOperand::OffsetType::IMMEDIATE;                  \
            addr.registerOffset.reset();                                           \
            addr.immediateOffset = immCtx->MINUS()                                 \
                ? ("-" + immCtx->IMMEDIATE()->getText())                           \
                : immCtx->IMMEDIATE()->getText();                                  \
        }                                                                          \
        operands.push_back(ptxemu::ir::OperandContext{addr});                                  \
        operands.push_back(createOperandFromContext(operandCtxs[1]));              \
        for (size_t i = 2; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) { \
            operands.push_back(createOperandFromContext(operandCtxs[i]));          \
        }                                                                          \
    } else {                                                                       \
        for (size_t i = 0; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) { \
            operands.push_back(createOperandFromContext(operandCtxs[i]));          \
        }                                                                          \
    }                                                                              \
                                                                                     \
    auto quals = extractQualifiersFromContext(ctx);                                   \
                                                                                       \
    /* Fix ambiguous qualifier mappings: tokenToQualifier matches first               \
     * occurrence in ptx_qualifier.def, but atom context requires the _ATOM           \
     * variant.  ".add" -> Q_DOTADD beats Q_ADD_ATOM;                                 \
     * ".or"  -> Q_DOTOR  beats Q_OR_ATOM.                                            \
     */                                                                                \
    for (auto &q : quals) {                                                            \
        switch (q) {                                                                   \
        case ptxemu::ir::Qualifier::Q_DOTADD: q = ptxemu::ir::Qualifier::Q_ADD_ATOM; break;                   \
        case ptxemu::ir::Qualifier::Q_DOTOR:  q = ptxemu::ir::Qualifier::Q_OR_ATOM;  break;                   \
        default: break;                                                                \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    auto stmtCtx = makeAtomInstr(quals, operands,                                      \
                                 (int)operands.size(),                                 \
                                 ctx->getText());                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                                \
                                                                                       \
    return nullptr;                                                                     \
}

// X-Macro展开
