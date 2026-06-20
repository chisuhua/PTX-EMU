// =============================================================================
// barrier.cpp - Warp-level Barrier Instruction Handlers
// =============================================================================
// @file barrier.cpp
// @brief Stage 3: bar.warp.sync 和 activemask 指令的实现
// @details 实现 PTX ISA v6.0+ 的 warp 级收敛屏障机制
//          使用 Wbar 数据结构实现线程同步
// @author PTX-EMU Team
// @date 2026-04-03
// 
// Stage 3 TODO:
// 1. ✅ BarWarpSyncHandler::processOperation() - 实现 warp 级屏障同步
// 2. ✅ ActivemaskHandler::processOperation() - 读取执行掩码
// =============================================================================

#include "ptxsim/instruction_handlers.h"
#include "ptxsim/barrier/barrier_module.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/wbar.h"
#include "ptxsim/sm_context.h"
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
// 3. Mark current thread as arrived using Wbar::arrive()
// 4. Check if all participants have arrived using Wbar::is_complete()
// 5. If complete:
//    - Update all participating threads' PC to reconvergence_pc
//    - Clear barrier for next use
// 6. If not complete:
//    - Mark thread as blocked (waiting at barrier)
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

    ptxsim::WarpState& warp_state = warp_ctx->get_warp_state();
    int lane_id = context->lane_id_;

    uint32_t current_pc = warp_state.threads[lane_id].pc;

    if (reconvergence_pc == 0 || reconvergence_pc == (int)current_pc) {
        reconvergence_pc = (int)current_pc + 1;
    }

    uint32_t dynamic_mask = 0;
    bool force_reconvergence_done = false;

    if (warp_state.current_wbar_id < 0) {
        for (int i = 0; i < 32; i++) {
            if (!warp_state.threads[i].is_active) continue;
            if (warp_state.threads[i].is_exited) continue;
            if (warp_state.threads[i].pc == current_pc ||
                warp_state.threads[i].next_pc == static_cast<uint32_t>(current_pc)) {
                dynamic_mask |= (1u << i);
            }
        }
    }

    auto unique_pcs = warp_ctx->get_unique_pcs();
    if (unique_pcs.size() > 1 && warp_state.current_wbar_id < 0) {
        warp_ctx->force_reconvergence_at_barrier(static_cast<int>(current_pc));

        warp_state.current_wbar_id = 0;
        ptxsim::Wbar& init_wbar = warp_state.wbars[0];

        SMContext* sm_ctx = warp_ctx->get_sm_context();

        // 在分歧路径中，部分线程不在 barrier PC，动态掩码不完整。
        // 应使用 PTX 指令的 static_mask 作为参与掩码，确保所有指定线程被计入。
        uint32_t participation_mask = static_mask;

        // BUG-RECONVERGENCE-SIMPLEGEMM fix:
        // If the wbar was already initialized (e.g., a divergent half of the warp
        // already passed through this barrier and was released), preserve its
        // arrived_mask instead of resetting it. Otherwise the first divergent
        // half's arrival record is lost, and subsequent arrivals can never
        // accumulate to the full participation_mask → barrier never completes
        // → lanes that arrive later are stuck at barrier_pc forever.
        if (!init_wbar.is_initialized) {
            init_wbar.init(participation_mask, reconvergence_pc);
        } else {
            init_wbar.participation_mask = participation_mask;
            init_wbar.reconvergence_pc = reconvergence_pc;
            init_wbar.expected_count = __builtin_popcount(participation_mask);
            init_wbar.is_initialized = true;
        }
        init_wbar.arrive(lane_id);

        if (init_wbar.is_complete() && warp_state.current_wbar_id >= 0) {
            warp_ctx->set_exec_mask(init_wbar.arrived_mask);
            for (int i = 0; i < WarpContext::WARP_SIZE; ++i) {
                if ((init_wbar.arrived_mask & (1u << i)) && warp_state.threads[i].is_active) {
                    warp_ctx->advance_thread_pc(i, reconvergence_pc);
                    warp_state.threads[i].is_blocked = false;
                    warp_state.threads[i].status = ptxsim::ThreadStatus::Active;
                }
            }
            // BUG-POSTBARRIER-TWOHALVES fix: OR with existing active_mask
            // to preserve lanes already released by a prior barrier call
            // (e.g. when a divergent warp hits the same barrier in two halves).
            warp_ctx->set_active_mask(
                warp_ctx->get_active_mask() | init_wbar.arrived_mask);
            warp_state.current_wbar_id = -1;
            set_pc_overridden(true);
        } else {
            warp_state.threads[lane_id].is_blocked = true;
            warp_state.threads[lane_id].status = ptxsim::ThreadStatus::Blocked;
            set_pc_overridden(true);
            PTX_DEBUG_THREAD("Lane %d blocked at forced reconvergence barrier (arrived=%d/%d)",
                            lane_id, init_wbar.count_arrived(), init_wbar.count_participants());
        }
        return;
    }

    if (force_reconvergence_done) {
        return;
    }

    int wbar_id = 0;
    ptxsim::Wbar& wbar = warp_state.wbars[wbar_id];

    if (warp_state.current_wbar_id < 0 && wbar.is_initialized) {
        wbar.reset();
    }

    if (!wbar.is_initialized) {
        uint32_t participation_mask = (dynamic_mask != 0) ? (dynamic_mask & static_mask) : static_mask;
        if (participation_mask == 0) participation_mask = static_mask;
        wbar.init(participation_mask, reconvergence_pc);
        warp_state.current_wbar_id = wbar_id;
        PTX_DEBUG_EMU("bar.warp.sync: Initialized wbar[%d] with mask=0x%X, reconvergence_pc=%d",
                      wbar_id, participation_mask, reconvergence_pc);
    }

    wbar.arrive(lane_id);

    // BUG-CUTE-RMSNORM-BROADCAST-SKIP: current_wbar_id < 0 means the wbar
    // was already released (current_wbar_id is set to -1 on release).
    // Re-checking is_complete() here would re-release the same lanes,
    // skipping the broadcast instruction at reconvergence_pc.
    if (wbar.is_complete() && warp_state.current_wbar_id >= 0) {
        if (reconvergence_pc < 0) {
            PTX_ERROR_EMU("bar.warp.sync: Invalid reconvergence_pc=%d at barrier completion, skipping PC update", reconvergence_pc);
            return;
        }

        warp_ctx->set_exec_mask(wbar.arrived_mask);

        PTX_INFO_EMU("bar.warp.sync: Barrier complete, releasing %d threads to PC=%d (mask=0x%X arrived=0x%X)",
                      wbar.count_participants(), reconvergence_pc,
                      wbar.participation_mask, wbar.arrived_mask);

        for (int i = 0; i < WarpContext::WARP_SIZE; ++i) {
            if ((wbar.arrived_mask & (1u << i)) && warp_state.threads[i].is_active) {
                uint32_t old_pc = warp_ctx->get_thread(i)->get_pc();
                warp_ctx->advance_thread_pc(i, reconvergence_pc);
                warp_state.threads[i].is_blocked = false;
                warp_state.threads[i].status = ptxsim::ThreadStatus::Active;
                PTX_INFO_EMU("  Released lane=%d: PC=%u -> %d", i, old_pc, reconvergence_pc);
            }
        }

        // BUG-POSTBARRIER-TWOHALVES fix: OR with existing active_mask
        // to preserve lanes already released by a prior barrier call.
        warp_ctx->set_active_mask(
            warp_ctx->get_active_mask() | wbar.arrived_mask);

        warp_state.current_wbar_id = -1;
        set_pc_overridden(true);
    } else {
        warp_state.threads[lane_id].is_blocked = true;
        warp_state.threads[lane_id].status = ptxsim::ThreadStatus::Blocked;
        set_pc_overridden(true);
        PTX_DEBUG_THREAD("Lane %d blocked at bar.warp.sync (arrived=%d/%d)",
                         lane_id, wbar.count_arrived(), wbar.count_participants());
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
        // (Mirrors legacy SMContext::synchronize_barrier at sm_context.cpp:703.)
        context->set_state(BAR_SYNC);
        context->set_next_pc(context->get_pc());
        PTX_DEBUG_THREAD("Thread [%u,%u,%u] waiting at barrier_id=%d",
                         context->ThreadIdx.x, context->ThreadIdx.y, context->ThreadIdx.z,
                         barId);
    }
}