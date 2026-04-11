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
    const BarWarpSyncInstr& instr = std::get<BarWarpSyncInstr>(stmt.data);
    processOperation(context, &(context->operand_collected[0]), instr.qualifiers,
                     &context->operand_is_immediate_);
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
    // Validate: bar.warp.sync should have 2 operands
    // Note: operands[] contains void* pointers to register/memory locations
    // For immediate operands, operand_is_immediate[i] is true and operands[i] points to the immediate value
    
    if (!operands || !operands[0] || !operands[1]) {
        PTX_ERROR_EMU("bar.warp.sync requires 2 operands");
        return;
    }
    
    // Step 1: Extract participation mask from operand[0]
    uint32_t participation_mask = 0;
    if (operand_is_immediate && (*operand_is_immediate)[0]) {
        // Immediate operand
        participation_mask = *static_cast<uint32_t*>(operands[0]);
    } else {
        // Register operand - operands[0] is the register address
        participation_mask = *static_cast<uint32_t*>(operands[0]);
    }
    
    // Step 2: Extract reconvergence PC from operand[1]
    // For bar.warp.sync, operand[1] is typically a label which has been resolved to PC
    int reconvergence_pc = -1;
    if (operand_is_immediate && (*operand_is_immediate)[1]) {
        // Immediate PC value
        reconvergence_pc = *static_cast<int*>(operands[1]);
    } else {
        // Label was resolved to PC during parsing
        reconvergence_pc = *static_cast<int*>(operands[1]);
    }
    
    // Step 3: Access WarpContext and WarpState
    WarpContext* warp_ctx = context->warp_context_;
    if (!warp_ctx) {
        PTX_ERROR_EMU("WarpContext is null in BarWarpSyncHandler");
        return;
    }
    
    ptxsim::WarpState& warp_state = warp_ctx->get_warp_state();
    int lane_id = context->lane_id_;
    
    // Step 4: Find or allocate a Wbar register
    // For now, use wbar_id = 0 (can be extended to support multiple barriers)
    int wbar_id = 0;
    ptxsim::Wbar& wbar = warp_state.wbars[wbar_id];
    
    // Step 5: Initialize barrier if not already initialized
    // Check if this is a new barrier by comparing participation masks
    if (!wbar.is_initialized || wbar.participation_mask != participation_mask) {
        // New barrier: initialize
        wbar.init(participation_mask, reconvergence_pc);
        warp_state.current_wbar_id = wbar_id;
        PTX_DEBUG_EMU("bar.warp.sync: Initialized wbar[%d] with mask=0x%X, reconvergence_pc=%d",
                      wbar_id, participation_mask, reconvergence_pc);
    }
    
    // Step 6: Mark current thread as arrived
    wbar.arrive(lane_id);
    PTX_DEBUG_THREAD("Lane %d arrived at bar.warp.sync (mask=0x%X, pc=%d)",
                     lane_id, participation_mask, reconvergence_pc);
    
    // Step 7: Check if all participants have arrived
    // For warp-level barriers with full participation mask, mark all lanes as arrived
    // but only update state for ACTIVE lanes
    if (participation_mask == 0xFFFFFFFF) {
        for (int i = 0; i < WarpContext::WARP_SIZE; ++i) {
            if (!(wbar.arrived_mask & (1u << i))) {
                wbar.arrive(i);
                warp_state.threads[i].is_blocked = false;
            }
        }
    }
    
    if (wbar.is_complete()) {
        PTX_DEBUG_EMU("bar.warp.sync: Barrier complete, releasing %d threads to PC=%d",
                      wbar.count_participants(), reconvergence_pc);
        
        // Only update ACTIVE lanes - inactive lanes remain at their current PC
        for (int i = 0; i < WarpContext::WARP_SIZE; ++i) {
            if ((wbar.participation_mask & (1u << i)) && warp_state.threads[i].is_active) {
                warp_ctx->set_thread_pc(i, reconvergence_pc);
                warp_ctx->update_pc_stack(i, reconvergence_pc);
                warp_state.threads[i].is_blocked = false;
            }
        }
        
        // Reset barrier for next use
        wbar.reset();
        warp_state.current_wbar_id = -1;
    } else {
        warp_state.threads[lane_id].is_blocked = true;
        warp_state.threads[lane_id].status = ptxsim::ThreadStatus::Blocked;
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
// Purpose: Synchronize all threads in a cooperative thread array (CTA/block)
// 
// Operation:
// 1. Extract barId from BarrierInstr (may be specified as operand or default to 0)
// 2. Get thread's warp context to access SM context
// 3. Call sm_context_->synchronize_barrier(barId, thread) to handle CTA-level sync
// 4. If all threads reach barrier, they are released to continue execution
// =============================================================================

void BarHandler::executeBarrier(ThreadContext* context, const BarrierInstr& instr) {
    // Step 1: Extract barId from BarrierInstr
    int barId = 0;  // Default barrier ID
    if (instr.barId.has_value()) {
        barId = instr.barId.value();
    }
    // For the type field, we expect this to indicate "cta" (cooperative thread array)
    // but we can proceed with the standard synchronize_barrier which works at CTA level
    
    // Step 2: Get SM context via warp context
    WarpContext* warp_ctx = context->get_warp_context();
    if (!warp_ctx) {
        PTX_ERROR_EMU("WarpContext is null in barHandler");
        context->next_pc = context->pc + 1;  // Advance PC to avoid infinite loop
        return;
    }
    
    SMContext* sm_context = warp_ctx->get_sm_context();  // Assuming this method exists
    if (!sm_context) {
        PTX_ERROR_EMU("SMContext is null in barHandler");
        context->next_pc = context->pc + 1;  // Advance PC to avoid infinite loop
        return;
    }
    
    PTX_DEBUG_THREAD("Thread [%u,%u,%u] executing bar.sync with barrier_id=%d",
                     context->ThreadIdx.x, context->ThreadIdx.y, context->ThreadIdx.z,
                     barId);
    
    // Step 3: Call synchronize_barrier to handle CTA-level synchronization
    PTX_INFO_EMU("Thread [%u,%u,%u] calling synchronize_barrier(barrier_id=%d)",
                 context->ThreadIdx.x, context->ThreadIdx.y, context->ThreadIdx.z,
                 barId);
    bool sync_complete = sm_context->synchronize_barrier(barId, context);
    
    if (sync_complete) {
        // All threads reached the barrier and have been released
        PTX_DEBUG_EMU("bar.sync complete: All threads released for barrier_id=%d", barId);
        context->next_pc = context->pc + 1;  // Advance PC after barrier
    } else {
        // Thread is still waiting at barrier - do not advance PC, will retry
        // Need to keep next_pc == pc so thread stays at current instruction
        context->next_pc = context->pc; 
        // The thread state is already set to BAR_SYNC by synchronize_barrier()
        PTX_DEBUG_THREAD("Thread [%u,%u,%u] waiting at barrier_id=%d",
                         context->ThreadIdx.x, context->ThreadIdx.y, context->ThreadIdx.z,
                         barId);
    }
}