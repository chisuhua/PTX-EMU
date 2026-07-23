// =============================================================================
// barrier.cpp - Warp-level Barrier Instruction Handlers
// =============================================================================
// @file barrier.cpp
// @brief Stage 3: bar.warp.sync 和 activemask 指令的实现
// @details 实现 PTX ISA v6.0+ 的 warp 级收敛屏障机制
//          通过 BarrierModule / WarpBarrier 实现线程同步
// @author PTX-EMU Team
// @date 2026-04-03
//
// 历史: Stage 3 全部子项已 ✅ 完成
//   1. ✅ BarWarpSyncHandler::processOperation() - 实现 warp 级屏障同步
//      (commit 0e31156 BarWarpSyncHandler 迁移)
//   2. ✅ ActivemaskHandler::processOperation() - 读取执行掩码
// =============================================================================

#include "ptxsim/instruction_handlers.h"
#include "ptxsim/barrier/barrier_module.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "utils/logger.h"
#include <cstdint>
#include <vector>
#include <memory>

// Bring ptxsim barrier types into global scope so the rest of this file
// (which lives in the global namespace) can reference them without prefix.
using ptxsim::BarrierModule;
using ptxsim::WarpBarrier;
using ptxsim::CTABarrier;

// Note: Handlers are defined in global namespace, not ptxsim namespace
// (to match the declarations in instruction_handlers.h)

// =============================================================================
// BarWarpSyncHandler Implementation
// =============================================================================
// PTX Syntax: bar.warp.sync mask, reconvergence_pc;
// 
// Purpose: Synchronize divergent threads within a warp
// 
// Operation:
// 1. Extract participation mask from operand[0]
// 2. Get reconvergence PC from operand[1] (label)
// 3. Route through BarrierModule::arrive_at_warp_barrier() to register
//    the participating lane and check completeness
// 4. If complete: update participating threads' PC to reconvergence_pc
// 5. If not complete: mark thread as blocked (waiting at barrier)
// =============================================================================

// Override prepareOperands to use BarWarpSyncInstr instead of GenericInstr
bool BarWarpSyncHandler::prepareOperands(ThreadContext* context, StatementContext& stmt) {
    BarWarpSyncInstr& instr = std::get<BarWarpSyncInstr>(stmt.data);
    
    // Defensive: ensure we have a data type qualifier
    // bar.warp.sync requires a data type (typically .b32 for the participation mask)
    // Check if qualifiers contains a data type qualifier (Q2bytes > 0)
    bool hasDataType = false;
    for (auto q : instr.qualifiers) {
        if (Q2bytes(q) > 0) {
            hasDataType = true;
            break;
        }
    }
    if (!hasDataType) {
        instr.qualifiers = {Qualifier::Q_B32};
    }
    
    if (!acquireAllOperands(context, instr.operands, instr.qualifiers, 
                           static_cast<int>(instr.operands.size()))) {
        return false;
    }
    context->collect_operands(stmt, instr.operands, &(instr.qualifiers));
    return true;
}

// Override executeOperation to use BarWarpSyncInstr
bool BarWarpSyncHandler::executeOperation(ThreadContext* context, StatementContext& stmt) {
    BarWarpSyncInstr& instr = std::get<BarWarpSyncInstr>(stmt.data);

    if (!acquireAllOperands(context, instr.operands, instr.qualifiers,
                           static_cast<int>(instr.operands.size()))) {
        return false;
    }
    context->collect_operands(stmt, instr.operands, &(instr.qualifiers));

    // Execute the barrier operation (may set next_pc to reconvergence_pc)
    processOperation(context, &(context->operand_collected[0]), instr.qualifiers,
                     &context->operand_is_immediate_);

    // Note: set_pc_overridden(true) is called INSIDE processOperation's else branch
    // when the thread is blocked. We do NOT call it here unconditionally.

    releaseAllOperands(instr.operands, static_cast<int>(instr.operands.size()));
    return true;
}

// Override commitResults to use BarWarpSyncInstr
bool BarWarpSyncHandler::commitResults(ThreadContext* context, StatementContext& stmt) {
    BarWarpSyncInstr& instr = std::get<BarWarpSyncInstr>(stmt.data);
    if (!instr.operands.empty()) {
        context->commit_operand(stmt, instr.operands[0], instr.qualifiers);
    }
    releaseAllOperands(instr.operands, static_cast<int>(instr.operands.size()));
    return true;
}

void BarWarpSyncHandler::processOperation(ThreadContext* context, void** operands,
                                          const std::vector<Qualifier>& qualifiers,
                                          const std::vector<char>* operand_is_immediate) {
    if (!operands || !operands[0] || !operands[1]) {
        PTX_ERROR_EMU("bar.warp.sync requires 2 operands");
        return;
    }

    uint32_t static_mask = *static_cast<uint32_t*>(operands[0]);

    int reconvergence_pc = *static_cast<int*>(operands[1]);

    WarpContext* warp_ctx = context->warp_context_;
    if (!warp_ctx) {
        PTX_ERROR_EMU("WarpContext is null in BarWarpSyncHandler");
        return;
    }

    CTAContext* cta_ctx = warp_ctx->get_cta_context();
    if (!cta_ctx) {
        PTX_ERROR_EMU("CTAContext is null in BarWarpSyncHandler — warp not linked to CTA");
        return;
    }
    BarrierModule& bm = cta_ctx->get_barrier_module();

    ptxsim::WarpState& warp_state = warp_ctx->get_warp_state();
    int lane_id = context->lane_id_;
    uint32_t current_pc = warp_state.threads[lane_id].pc;

    if (reconvergence_pc == 0 || reconvergence_pc == (int)current_pc) {
        reconvergence_pc = (int)current_pc + 1;
    }

    // Find the appropriate warp barrier slot:
    //   1. First, look for an existing INITIALIZED barrier at the same
    //      barrier_pc (reuse — other lanes may have already arrived)
    //   2. If none found, allocate the first UNINITIALIZED slot
    //
    // This preserves the single-barrier semantics of bar.warp.sync while
    // supporting 16 slots for future multi-barrier CUTLASS scenarios.
    // Without step 1, each lane's arrival allocates a different slot
    // and the barrier never completes.
    int wbar_id = 0;
    WarpBarrier* wbar = nullptr;
    for (int i = 0; i < BarrierModule::MAX_WARP_BARRIERS; ++i) {
        WarpBarrier* candidate = bm.get_warp_barrier(i);
        if (candidate && candidate->is_initialized() &&
            candidate->get_barrier_pc() == current_pc) {
            wbar_id = i;
            wbar = candidate;
            break;
        }
    }
    if (!wbar) {
        for (int i = 0; i < BarrierModule::MAX_WARP_BARRIERS; ++i) {
            WarpBarrier* candidate = bm.get_warp_barrier(i);
            if (!candidate || !candidate->is_initialized()) {
                wbar_id = i;
                wbar = bm.get_warp_barrier(wbar_id);
                break;
            }
        }
    }

    uint32_t dynamic_mask = 0;
    bool wbar_was_initialized = (wbar != nullptr && wbar->is_initialized());

    if (!wbar_was_initialized) {
        for (int i = 0; i < 32; i++) {
            if (!warp_state.threads[i].is_active) continue;
            if (warp_state.threads[i].is_exited) continue;
            if (warp_state.threads[i].pc == current_pc ||
                warp_state.threads[i].next_pc == static_cast<uint32_t>(current_pc)) {
                dynamic_mask |= (1u << i);
            }
        }
    }

    // ── Path A: force_reconvergence (divergent scenario) ──
    auto unique_pcs = warp_ctx->get_unique_pcs();
    if (unique_pcs.size() > 1 && !wbar_was_initialized) {
        warp_ctx->force_reconvergence_at_barrier(static_cast<int>(current_pc));

        bm.init_warp_barrier(wbar_id, static_mask, reconvergence_pc, current_pc);
        wbar = bm.get_warp_barrier(wbar_id);
        wbar->arrive(lane_id);

        if (wbar->is_complete()) {
            bm.release_warp_barrier(wbar_id, warp_ctx);
            set_pc_overridden(true);
        } else {
            warp_state.threads[lane_id].is_blocked = true;
            warp_state.threads[lane_id].status = ptxsim::ThreadStatus::Blocked;
            set_pc_overridden(true);
            PTX_DEBUG_THREAD("Lane %d blocked at forced reconvergence barrier (arrived=%d/%d)",
                            lane_id, wbar->get_arrived_count(), wbar->get_expected_count());
        }
        return;
    }

    // ── Path B: normal barrier (no divergence) ──
    if (!wbar->is_initialized()) {
        uint32_t participation_mask = (dynamic_mask != 0) ? (dynamic_mask & static_mask) : static_mask;
        if (participation_mask == 0) participation_mask = static_mask;
        bm.init_warp_barrier(wbar_id, participation_mask, reconvergence_pc, current_pc);
        wbar = bm.get_warp_barrier(wbar_id);
        PTX_DEBUG_EMU("bar.warp.sync: Initialized wbar[%d] with mask=0x%X, reconvergence_pc=%d",
                      wbar_id, participation_mask, reconvergence_pc);
    }

    wbar->arrive(lane_id);

    if (wbar->is_complete()) {
        if (reconvergence_pc < 0) {
            PTX_ERROR_EMU("bar.warp.sync: Invalid reconvergence_pc=%d at barrier completion, skipping PC update", reconvergence_pc);
            return;
        }

        PTX_INFO_EMU("bar.warp.sync: Barrier complete, releasing %d threads to PC=%d (mask=0x%X arrived=0x%X)",
                      wbar->get_expected_count(), reconvergence_pc,
                      wbar->get_participation_mask(), wbar->get_arrived_mask());

        bm.release_warp_barrier(wbar_id, warp_ctx);
        set_pc_overridden(true);
    } else {
        warp_state.threads[lane_id].is_blocked = true;
        warp_state.threads[lane_id].status = ptxsim::ThreadStatus::Blocked;
        set_pc_overridden(true);
        PTX_DEBUG_THREAD("Lane %d blocked at bar.warp.sync (arrived=%d/%d)",
                         lane_id, wbar->get_arrived_count(), wbar->get_expected_count());
    }
}

// =============================================================================
// ActivemaskHandler Implementation
// =============================================================================
// PTX Syntax: activemask.b32 dst;
// 
// Purpose: Read the current active lane mask into a destination register
// 
// Operation:
// 1. Access WarpContext to get exec_mask
// 2. Write exec_mask to destination register
// =============================================================================

void ActivemaskHandler::processOperation(ThreadContext* context, void** operands,
                                         const std::vector<Qualifier>& qualifiers,
                                         const std::vector<char>* operand_is_immediate) {
    // Activemask has 1 operand: destination register
    
    if (!operands || !operands[0]) {
        PTX_ERROR_EMU("Activemask: null operands");
        return;
    }
    
    // Access WarpContext to get exec_mask
    WarpContext* warp_ctx = context->warp_context_;
    if (!warp_ctx) {
        PTX_ERROR_EMU("WarpContext is null in ActivemaskHandler");
        return;
    }
    
    // Read current exec_mask from WarpState
    uint32_t exec_mask = warp_ctx->get_exec_mask();
    
    // Write to destination register
    uint32_t* dst_reg = static_cast<uint32_t*>(operands[0]);
    *dst_reg = exec_mask;
    
    PTX_DEBUG_THREAD("activemask.b32: read exec_mask=0x%X to reg=%p", exec_mask, dst_reg);
}

// =============================================================================
// barHandler Implementation for S_BAR (bar.sync)
// =============================================================================
// PTX Syntax: bar.sync [cta,] [barrier_id];
//
// Purpose: Synchronize all threads in a cooperative thread array (CTA/block).
// State is managed by BarrierModule (owned by CTAContext); this handler is a
// thin dispatcher that calls arrive_at_cta_barrier + release_cta_barrier.
//
// Operation:
// 1. Extract barId from BarrierInstr (operand or default to 0)
// 2. Get WarpContext → CTAContext via reverse link → get BarrierModule
// 3. Call bm.arrive_at_cta_barrier(barId, thread)
// 4. On complete, call bm.release_cta_barrier(barId, cta_ctx, post_pc)
//    which advances every arrived thread's per-thread PC.
// =============================================================================

void BarHandler::executeBarrier(ThreadContext* context, const BarrierInstr& instr) {
    int barId = 0;
    if (instr.barId.has_value()) {
        barId = instr.barId.value();
    }

    WarpContext* warp_ctx = context->get_warp_context();
    if (!warp_ctx) {
        PTX_ERROR_EMU("WarpContext is null in barHandler");
        context->set_next_pc(context->get_pc() + 1);
        return;
    }

    CTAContext* cta_ctx = warp_ctx->get_cta_context();
    if (!cta_ctx) {
        PTX_ERROR_EMU("CTAContext is null in barHandler — warp not linked to CTA");
        context->set_next_pc(context->get_pc() + 1);
        return;
    }

    PTX_DEBUG_THREAD("Thread [%u,%u,%u] executing bar.sync with barrier_id=%d",
                     context->ThreadIdx.x, context->ThreadIdx.y, context->ThreadIdx.z,
                     barId);

    if (warp_ctx->get_unique_pcs().size() > 1) {
        warp_ctx->force_reconvergence_at_barrier(context->get_pc());
    }

    BarrierModule& bm = cta_ctx->get_barrier_module();
    bool sync_complete = bm.arrive_at_cta_barrier(barId, context);

    if (sync_complete) {
        int post_barrier_pc = context->get_pc() + 1;
        bm.release_cta_barrier(barId, cta_ctx, post_barrier_pc);
        PTX_DEBUG_EMU("bar.sync complete: All threads released for barrier_id=%d -> PC=%d",
                      barId, post_barrier_pc);
        context->set_next_pc(post_barrier_pc);
    } else {
        // Mark thread as waiting at barrier so the executor (warp_context.cpp:267)
        // recognizes BAR_SYNC and skips re-execution. Without this, sync_to_warp_state()
        // keeps is_blocked=false and the scheduler spins on the barrier instruction.
        context->set_state(BAR_SYNC);
        context->set_next_pc(context->get_pc());
        PTX_DEBUG_THREAD("Thread [%u,%u,%u] waiting at barrier_id=%d",
                         context->ThreadIdx.x, context->ThreadIdx.y, context->ThreadIdx.z,
                         barId);
    }
}