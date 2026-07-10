// src/ptxsim/instructions/tcgen05_fence.cpp
// Phase 4 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q6-B).
//
// tcgen05.fence is a NO-OP MARKER per Oracle Q6-B + design D8:
//   - No membar / FENCE memory barrier
//   - No WarpBarrier interaction (no warp arrival blocking)
//   - No CTAContext / TmemAllocator / Smem side effects
//   - PC advancement handled by dispatch wrapper (Tcgen05Handler::processTcgen05Operation)
//   - Only side effect: warp->record_fence_position(before/after)
//
// Per ptx-lessons-learned §2: no mutex needed — verified by grep on
// warp_context.cpp + warp_state.h returning zero lock_guard/unique_lock/std::mutex
// matches on the warp state path. Single-writer (warp scheduler) by construction.
//
// UNVERIFIED-AGAINST-HARDWARE — exact semantics of fence marker in future
// hardware memory-model implementations are not validated (extension point only).

#include "ptxsim/instructions/tcgen05.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "utils/logger.h"

#include <cstdint>
#include <stdexcept>

namespace ptxsim {

namespace {
// Qualifier scan helpers — same pattern as Phase 3 (tcgen05.cpp is_ws_path /
// has_f16_qualifier). Linear scan over enum-vector Qualifier; explicit specialization
// since Q_BEFORE_THREAD_SYNC / Q_AFTER_THREAD_SYNC come from grammar tokens that the
// visitor stores in instr.qualifiers.
bool has_before_sync_qualifier(const Tcgen05Instr &instr) {
    for (auto q : instr.qualifiers) {
        if (q == Qualifier::Q_BEFORE_THREAD_SYNC) return true;
    }
    return false;
}
bool has_after_sync_qualifier(const Tcgen05Instr &instr) {
    for (auto q : instr.qualifiers) {
        if (q == Qualifier::Q_AFTER_THREAD_SYNC) return true;
    }
    return false;
}
}  // namespace

void processTcgen05Fence(ThreadContext *context, const Tcgen05Instr &instr) {
    WarpContext *warp = context->get_warp_context();
    if (!warp) {
        PTX_ERROR_EMU("tcgen05.fence: no WarpContext attached to thread");
        throw UnsupportedInstructionException(
            "tcgen05.fence", "tcgen05.fence requires an active WarpContext");
    }

    // Oracle Q2-A consistency: 11/11 handlers reject cta_group::2 with the
    // same ADR-0018 message even though fence semantic is cluster-agnostic.
    // Keeps spec scenario "cta_group::2 throws clear exception" uniform.
    if (instr.cta_group == 2) {
        throw_cta_group_2("tcgen05.fence");
    }

    // PTX ISA §9.7.16 grammar: tcgen05.fence MUST have exactly one of
    // ::before_thread_sync / ::after_thread_sync. Grammar rejects both/neither
    // at parse; runtime sanity check covers hand-constructed Tcgen05Instr
    // (used by unit tests and oracle-style construction in step_warp harness).
    const bool before = has_before_sync_qualifier(instr);
    const bool after  = has_after_sync_qualifier(instr);
    if (before == after) {  // both true OR both false
        PTX_ERROR_EMU(
            "tcgen05.fence: must have exactly one of before/after_thread_sync");
        throw UnsupportedInstructionException(
            "tcgen05.fence",
            "tcgen05.fence requires exactly one of ::before_thread_sync or "
            "::after_thread_sync (PTX ISA §9.7.16). Both or neither is invalid.");
    }

    // Only side effect: record fence position. No membar, no WarpBarrier,
    // no active_mask mutation, no PC mutation. Dispatch wrapper advances PC.
    const WarpContext::FencePosition pos =
        before ? WarpContext::kFenceBefore : WarpContext::kFenceAfter;
    warp->record_fence_position(pos);

    PTX_DEBUG_EMU("tcgen05.fence::%s_thread_sync (no-op marker, recorded)",
                  before ? "before" : "after");
}

}  // namespace ptxsim
