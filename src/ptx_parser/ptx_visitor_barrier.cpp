// BARRIER 类别的实现
// Stage 4: Add translation layer for bar.sync -> bar.warp.sync

#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/kernel_context.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_factory.h"
#include "utils/logger.h"

using namespace ptxir::factory;

namespace {

// Compute actual CTA thread count with fallback chain.
// Returns the number capped at 32 (warp size).
int compute_actual_thread_count(KernelContext* kernel) {
    if (!kernel) return 32;
    int total = 1;
    if (kernel->reqntid.x > 0 || kernel->reqntid.y > 0 || kernel->reqntid.z > 0) {
        total = kernel->reqntid.x * kernel->reqntid.y * kernel->reqntid.z;
    } else if (kernel->maxntid.x > 0 && kernel->maxntid.y > 0 && kernel->maxntid.z > 0) {
        total = kernel->maxntid.x * kernel->maxntid.y * kernel->maxntid.z;
    }
    return std::min(total, 32);
}

// Helper: Check if this is a warp-level barrier (single warp CTA)
// Returns true if CTA has <= 32 threads (1 warp)
bool isWarpLevelBarrier(KernelContext* kernel) {
    if (!kernel) return false;
    // Only translate when thread count is explicitly specified in the PTX source.
    // If neither .reqntid nor .maxntid directives are present, the thread count
    // is unknown at parse time and the decision must be deferred to the
    // interpreter which has access to the launch-time blockDim.
    // (e.g., cute_rmsnorm with blockDim=256 has no .reqntid in PTX,
    // yet the visitor would incorrectly translate bar.sync to bar.warp.sync
    // and break multi-warp CTA synchronization.)
    bool has_known_thread_count =
        (kernel->reqntid.x > 0 || kernel->reqntid.y > 0 || kernel->reqntid.z > 0) ||
        (kernel->maxntid.x > 0 && kernel->maxntid.y > 0 && kernel->maxntid.z > 0);
    if (!has_known_thread_count) return false;
    return compute_actual_thread_count(kernel) <= 32;
}
} // anonymous namespace

// VISITOR_BARRIER macro for X-macro in ptx_visitor.cpp (handles regular bar.sync, bar.arrive, etc.)
// Stage 4: Translate bar.sync to bar.warp.sync for single-warp CTAs
//
// TRANSLATION LOGIC (per architecture doc sm90_100.md:294):
//   - bar.sync is a CTA-LEVEL barrier that forces reconvergence of threads from different warps
//   - "bar.sync 0 — Block 级全同步。未汇合的 Warp 会在此被强制汇合"
//   - For multi-warp CTAs: MUST keep as S_BAR (bar.sync) because CTA-level forced reconvergence
//     behavior is required - warps that haven't reconverged will be forced to at this barrier
//   - For single-warp CTAs: OPTIMIZATION - translate to bar.warp.sync because there's only one warp
//     so CTA-level and warp-level semantics are equivalent, and bar.warp.sync is more efficient
//
// IMPORTANT: bar.warp.sync is an INTERNAL instruction (not real PTX ISA) - used for optimization
// when we can prove that CTA-level and warp-level barrier behavior are semantically equivalent.
#define VISITOR_BARRIER(openum, opstr, opname, opcount) \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) { \
    if (!currentKernel) return nullptr; \
    \
    /* Stage 4 Translation: Check if this is bar.sync and should be translated to bar.warp.sync */ \
    if (openum == S_BAR && isWarpLevelBarrier(currentKernel)) { \
        int total_threads = compute_actual_thread_count(currentKernel); \
        uint32_t mask = (total_threads >= 32) ? 0xFFFFFFFFu : ((1u << total_threads) - 1); \
        \
        std::vector<ptxemu::ir::OperandContext> operands; \
        operands.push_back(ptxemu::ir::OperandContext{ImmOperand{std::to_string(mask)}}); \
        operands.push_back(ptxemu::ir::OperandContext{ImmOperand{"0"}}); \
        \
        auto stmtCtx = makeBarWarpSyncInstr({ptxemu::ir::Qualifier::Q_B32}, operands,      \
            "bar.warp.sync.b32 " + std::to_string(mask) + ", 0;");           \
        currentKernel->kernelStatements.push_back(stmtCtx); \
        \
        return nullptr; \
    } \
    \
    /* Original bar.sync handling for multi-warp CTAs */ \
    std::optional<int> barId; \
    if (ctx->barrierOperands() && ctx->barrierOperands()->IMMEDIATE()) { \
        barId = extractIntFromToken(ctx->barrierOperands()->IMMEDIATE()->getSymbol()); \
    } \
    \
    auto stmtCtx = makeBarrierInstr(openum, extractQualifiersFromContext(ctx), \
                                   barId, "", ctx->getText());               \
    currentKernel->kernelStatements.push_back(stmtCtx); \
    \
    return nullptr; \
}

// VISITOR_WARP_BARRIER: Manual implementation of bar.warp.sync instruction
std::any PtxVisitor::visitBarWarpSyncInst(ptxparser::ptxParser::BarWarpSyncInstContext *ctx) {
    if (!currentKernel) return nullptr;

    std::vector<ptxemu::ir::Qualifier> qualifiers = extractQualifiersFromContext(ctx);

    bool hasDataType = false;
    for (auto q : qualifiers) {
        if (Q2bytes(q) > 0) {
            hasDataType = true;
            break;
        }
    }
    if (!hasDataType) {
        qualifiers = {ptxemu::ir::Qualifier::Q_B32};
        PTX_WARN("bar.warp.sync missing data type qualifier, defaulting to .b32");
    }

    std::vector<ptxemu::ir::OperandContext> operands;
    auto operandCtxs = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();
    if (!operandCtxs.empty()) {
        operands.push_back(createOperandFromContext(operandCtxs[0]));
    }

    auto stmtCtx = makeBarWarpSyncInstr(qualifiers, operands, ctx->getText());
    currentKernel->kernelStatements.push_back(stmtCtx);

    return nullptr;
}
