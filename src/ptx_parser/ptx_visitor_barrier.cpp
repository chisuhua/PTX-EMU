// BARRIER 类别的实现
// Stage 4: Add translation layer for bar.sync -> bar.warp.sync

#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/kernel_context.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "utils/logger.h"
#include <algorithm>
#include <any>
#include <cstdio>

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
    return compute_actual_thread_count(kernel) <= 32;
}
} // anonymous namespace

// VISITOR_BARRIER macro for X-macro in ptx_visitor.cpp (handles regular bar.sync, bar.arrive, etc.)
// Stage 4: Translate bar.sync to bar.warp.sync for single-warp CTAs
#define VISITOR_BARRIER(openum, opstr, opname, opcount) \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) { \
    if (!currentKernel) return nullptr; \
    \
    /* Stage 4 Translation: Check if this is bar.sync and should be translated to bar.warp.sync */ \
    if (openum == S_BAR && isWarpLevelBarrier(currentKernel)) { \
        int total_threads = compute_actual_thread_count(currentKernel); \
        uint32_t mask = (total_threads >= 32) ? 0xFFFFFFFFu : ((1u << total_threads) - 1); \
        \
        /* Translate to bar.warp.sync */ \
        StatementContext stmtCtx; \
        stmtCtx.instructionText = "bar.warp.sync.b32 " + std::to_string(mask) + ", " + std::to_string(currentKernel->kernelStatements.size() + 1) + ";"; \
        stmtCtx.type = S_BAR_WARP_SYNC; \
        \
        BarWarpSyncInstr instr; \
        instr.qualifiers = {Qualifier::Q_B32}; \
        \
        /* Also set StatementContext's qualifier field for consistency */ \
        stmtCtx.qualifier = {Qualifier::Q_B32}; \
        \
        /* Create operand for participation mask (dynamically computed from CTA thread count) */ \
        OperandContext maskOperand{ImmOperand{std::to_string(mask)}}; \
        instr.operands.push_back(maskOperand); \
        \
        /* Initialize reconvergence PC to -1 (unknown) - will be updated by CFG analysis */ \
        /* CFG post-dominator analysis in ptx_interpreter.cpp will set the correct value */ \
        int placeholder_pc = -1; \
        OperandContext pcOperand{ImmOperand{std::to_string(placeholder_pc)}}; \
        instr.operands.push_back(pcOperand); \
        \
        /* Reconvergence label placeholder (actual PC comes from CFG analysis) */ \
        instr.reconvergenceLabel = ""; \
        \
        stmtCtx.data = instr; \
        currentKernel->kernelStatements.push_back(stmtCtx); \
        \
        return nullptr; \
    } \
    \
    /* Original bar.sync handling for multi-warp CTAs */ \
    StatementContext stmtCtx; \
    stmtCtx.instructionText = ctx->getText(); \
    stmtCtx.type = openum; \
    \
    BarrierInstr instr; \
    instr.qualifiers = extractQualifiersFromContext(ctx); \
    \
    /* Extract bar ID from barrierOperands if present */ \
    if (ctx->barrierOperands() && ctx->barrierOperands()->IMMEDIATE()) { \
        instr.barId = extractIntFromToken(ctx->barrierOperands()->IMMEDIATE()->getSymbol()); \
    } \
    \
    stmtCtx.data = instr; \
    currentKernel->kernelStatements.push_back(stmtCtx); \
    \
    return nullptr; \
}

// VISITOR_WARP_BARRIER: Manual implementation of bar.warp.sync instruction
std::any PtxVisitor::visitBarWarpSyncInst(ptxparser::ptxParser::BarWarpSyncInstContext *ctx) {
    if (!currentKernel) return nullptr;
    
    StatementContext stmtCtx;
    stmtCtx.instructionText = ctx->getText();
    stmtCtx.type = S_BAR_WARP_SYNC;
    
    BarWarpSyncInstr instr;
    instr.qualifiers = extractQualifiersFromContext(ctx);
    
    // Defensive: ensure we have a data type qualifier
    // bar.warp.sync requires a data type (typically .b32 for the participation mask)
    bool hasDataType = false;
    for (auto q : instr.qualifiers) {
        if (Q2bytes(q) > 0) {
            hasDataType = true;
            break;
        }
    }
    if (!hasDataType) {
        instr.qualifiers = {Qualifier::Q_B32};
        PTX_WARN("bar.warp.sync missing data type qualifier, defaulting to .b32");
    }
    
    // Parse participation mask operand
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>();
    if (!operands.empty()) {
        instr.operands.push_back(createOperandFromContext(operands[0]));
    }
    
    // Parse reconvergence label
    if (ctx->labelOperand()) {
        instr.reconvergenceLabel = ctx->labelOperand()->ID()->getText();
    }
    
    stmtCtx.data = instr;
    currentKernel->kernelStatements.push_back(stmtCtx);
    
    return nullptr;
}
