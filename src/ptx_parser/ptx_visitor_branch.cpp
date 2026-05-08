#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/statement_factory.h"
#include "utils/logger.h"

using namespace ptxir::factory;

#define VISITOR_BRANCH(openum, opstr, opname, opcount)                                 \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                                  \
    std::string target;                                                           \
    if (ctx->labelOperand()) {                                                    \
        target = ctx->labelOperand()->getText();                                  \
        if (!target.empty() && target[0] == '$') {                                \
            target = target.substr(1);                                            \
        }                                                                          \
    }                                                                              \
                                                                                  \
    std::string predicate;                                                         \
    bool predicate_negated = false;                                                \
    if (ctx->predicate()) {                                                       \
        if (ctx->predicate()->BANG()) {                                           \
            predicate_negated = true;                                             \
        }                                                                         \
        if (ctx->predicate()->operand()) {                                        \
            predicate = ctx->predicate()->operand()->getText();                   \
        }                                                                         \
    }                                                                              \
                                                                                  \
    auto stmtCtx = makeBranchInstr(openum, extractQualifiersFromContext(ctx),     \
                                   target, predicate, predicate_negated,            \
                                   ctx->getText());                               \
    currentKernel->kernelStatements.push_back(stmtCtx);                            \
                                                                                  \
    return nullptr;                                                               \
}

