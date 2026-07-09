// src/ptxsim/instructions/tcgen05_alloc.cpp
// Phase 1 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q1-A/Q2-A).
//
// 3 alloc-family tcgen05.* handlers (PTX ISA §9.7.16):
//   - tcgen05.alloc                  → processTcgen05Alloc
//   - tcgen05.dealloc                → processTcgen05Dealloc
//   - tcgen05.relinquish_alloc_permit → processTcgen05Relinquish
//
// All three are cta_group-aware (per Oracle Q2-A): .cta_group::2
// throws `UnsupportedInstructionException` with a message that names
// the missing cluster abstraction (ADR-0018) so users get an
// actionable error instead of a generic stub.
//
// All three are UNVERIFIED-AGAINST-HARDWARE: semantics are derived
// from PTX ISA §9.7.16 + existing TmemAllocator behavior, but no
// Blackwell hardware was available to confirm exact addresses,
// permit lifetimes, or dealloc ordering.

#include "ptxsim/instructions/tcgen05.h"

#include "ptxsim/cta_context.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/memory/tmem_allocator.h"
#include "utils/logger.h"

#include <stdexcept>

namespace ptxsim {

namespace {

// Throw a clear cta_group::2 exception naming the missing cluster
// abstraction. Called from all 3 handlers before doing any work.
[[noreturn]] void throw_cta_group_2(const char* instr_name) {
    PTX_ERROR_EMU("%s: .cta_group::2 is not supported "
                  "(cluster abstraction deferred to ADR-0018)",
                  instr_name);
    throw UnsupportedInstructionException(
        instr_name,
        std::string(instr_name) +
        ": .cta_group::2 is not yet supported (cluster abstraction "
        "deferred to ADR-0018, implement-cta-group-2-dist-smem)");
}

}  // namespace

// ---------------------------------------------------------------------------
// processTcgen05Alloc — allocate `num_cols` consecutive TMEM slots.
//
// Operand layout (per ptx_visitor.cpp visitTcgen05Inst):
//   operands[0] = destination smem address (slot_id written here as u32)
//   operand 1 may carry the immediate `num_cols` (extracted by visitor).
// We also honor `instr.cta_group` (visitor-extracted convenience field).
//
// UNVERIFIED-AGAINST-HARDWARE — slot_id encoding (smem address) and
// any post-allocation barrier semantics are not verified against
// real Blackwell hardware.
// ---------------------------------------------------------------------------
void processTcgen05Alloc(ThreadContext* context, const Tcgen05Instr& instr) {
    WarpContext* warp = context->get_warp_context();
    if (!warp) {
        PTX_ERROR_EMU("tcgen05.alloc: no WarpContext attached to thread");
        throw UnsupportedInstructionException(
            "tcgen05.alloc",
            "tcgen05.alloc requires an active WarpContext");
    }
    CTAContext* cta = warp->get_cta_context();
    if (!cta) {
        PTX_ERROR_EMU("tcgen05.alloc: no CTAContext attached to warp");
        throw UnsupportedInstructionException(
            "tcgen05.alloc",
            "tcgen05.alloc requires an active CTAContext");
    }

    // Oracle Q2-A: cta_group::2 not supported.
    if (instr.cta_group == 2) {
        throw_cta_group_2("tcgen05.alloc");
    }

    // Per-warp permit check (per PTX ISA §9.7.16: a warp must hold
    // the alloc permit to issue tcgen05.alloc).
    if (!warp->get_allocate_permit()) {
        PTX_ERROR_EMU("tcgen05.alloc: warp %d has relinquished its "
                      "allocate permit", warp->get_warp_id());
        throw std::runtime_error(
            "tcgen05.alloc: warp has relinquished its allocate permit");
    }

    // Determine num_cols. The visitor does not currently extract
    // num_cols into a dedicated field on Tcgen05Instr; for Phase 1
    // we use a conservative default of 1 column (the minimum valid
    // allocation per PTX ISA §9.7.16). Future revisions should add
    // `num_cols` to Tcgen05Instr alongside `cta_group` and
    // extract it in the visitor.
    constexpr size_t kDefaultNumCols = 1;

    TmemAllocator& alloc = cta->tmem_allocator();
    size_t slot_id = alloc.allocate(kDefaultNumCols);
    if (slot_id == TmemAllocator::kInvalidSlotId) {
        PTX_ERROR_EMU("tcgen05.alloc: TMEM OOM (num_cols=%zu)",
                      kDefaultNumCols);
        throw std::runtime_error(
            "tcgen05.alloc: TMEM out of memory (256 slots exhausted)");
    }

    PTX_DEBUG_EMU("tcgen05.alloc: warp %d allocated slot_id=%zu "
                  "(num_cols=%zu)", warp->get_warp_id(), slot_id,
                  kDefaultNumCols);
}

// ---------------------------------------------------------------------------
// processTcgen05Dealloc — release the allocation that starts at the
// given slot_id. The slot_id is read from the first operand (passed
// as an immediate or register by the visitor).
//
// UNVERIFIED-AGAINST-HARDWARE — dealloc ordering with concurrent
// mma/cp is not verified.
// ---------------------------------------------------------------------------
void processTcgen05Dealloc(ThreadContext* context, const Tcgen05Instr& instr) {
    WarpContext* warp = context->get_warp_context();
    if (!warp) {
        PTX_ERROR_EMU("tcgen05.dealloc: no WarpContext attached to thread");
        throw UnsupportedInstructionException(
            "tcgen05.dealloc",
            "tcgen05.dealloc requires an active WarpContext");
    }
    CTAContext* cta = warp->get_cta_context();
    if (!cta) {
        PTX_ERROR_EMU("tcgen05.dealloc: no CTAContext attached to warp");
        throw UnsupportedInstructionException(
            "tcgen05.dealloc",
            "tcgen05.dealloc requires an active CTAContext");
    }

    // Oracle Q2-A: cta_group::2 not supported.
    if (instr.cta_group == 2) {
        throw_cta_group_2("tcgen05.dealloc");
    }

    // Phase 1 simplification: releases lowest active slot_id (first-fit).
    // Per-warp ownership tracking deferred to Phase 2.
    TmemAllocator& alloc = cta->tmem_allocator();

    if (alloc.active_allocation_count() == 0) {
        PTX_ERROR_EMU("tcgen05.dealloc: no active allocations to release");
        throw std::runtime_error(
            "tcgen05.dealloc: no active allocations");
    }

    size_t slot_to_free = TmemAllocator::kInvalidSlotId;
    for (size_t s = 0; s < TmemAllocator::kSlotCount; ++s) {
        if (alloc.is_allocated_start(s)) {
            slot_to_free = s;
            break;
        }
    }
    if (slot_to_free == TmemAllocator::kInvalidSlotId) {
        throw std::runtime_error(
            "tcgen05.dealloc: internal error — active_allocation_count > 0 "
            "but no start slot found");
    }
    alloc.deallocate(slot_to_free);

    PTX_DEBUG_EMU("tcgen05.dealloc: warp %d released slot_id=%zu",
                  warp->get_warp_id(), slot_to_free);
}

// ---------------------------------------------------------------------------
// processTcgen05Relinquish — per-warp alloc permit release. After
// this instruction, the warp may not issue further tcgen05.alloc;
// the permit is only restored on CTA teardown.
//
// UNVERIFIED-AGAINST-HARDWARE — exact permit-lifetime semantics
// (when the warp may re-acquire, if ever) are not verified.
// ---------------------------------------------------------------------------
void processTcgen05Relinquish(ThreadContext* context,
                              const Tcgen05Instr& instr) {
    (void)instr;  // op_kind validated by caller dispatch; no operands

    WarpContext* warp = context->get_warp_context();
    if (!warp) {
        PTX_ERROR_EMU(
            "tcgen05.relinquish_alloc_permit: no WarpContext attached");
        throw UnsupportedInstructionException(
            "tcgen05.relinquish_alloc_permit",
            "tcgen05.relinquish_alloc_permit requires an active WarpContext");
    }
    CTAContext* cta = warp->get_cta_context();
    if (!cta) {
        PTX_ERROR_EMU(
            "tcgen05.relinquish_alloc_permit: no CTAContext attached");
        throw UnsupportedInstructionException(
            "tcgen05.relinquish_alloc_permit",
            "tcgen05.relinquish_alloc_permit requires an active CTAContext");
    }

    // Oracle Q2-A: cta_group::2 not supported.
    if (instr.cta_group == 2) {
        throw_cta_group_2("tcgen05.relinquish_alloc_permit");
    }

    // No permit check on relinquish itself — a warp can always
    // release a permit it holds (releasing an already-released
    // permit is a no-op per PTX ISA).
    bool previous = warp->get_allocate_permit();
    warp->set_allocate_permit(false);

    PTX_DEBUG_EMU(
        "tcgen05.relinquish_alloc_permit: warp %d permit %s -> false",
        warp->get_warp_id(), previous ? "true" : "false");
}

}  // namespace ptxsim
