// src/ptxsim/instructions/tcgen05_cp.cpp
// Phase 2 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q4-B/Q2-A).
//
// tcgen05.cp — copy data from per-CTA shared memory to TMEM (PTX ISA
// §9.7.16). Distinct from tcgen05.ld which uses a TMA descriptor and
// reads from global memory; cp is the explicit "shared → TMEM" path
// used by weight-stationary tcgen05.mma.ws pipelines.
//
// Per Oracle Q4-B: reuse `cta->sharedMemSpace` (the raw per-CTA shared
// memory backing store) plus the existing `SharedMemoryManager` for
// bounds checking — DO NOT introduce a new `SmemDescriptor` abstraction
// (per `ptx_op.def:132` operand count is 3: dst, src, size — no
// descriptor field).
//
// Per Oracle Q2-A: `.cta_group::2` throws `UnsupportedInstructionException`
// with a message that names the missing cluster abstraction (ADR-0018).
//
// UNVERIFIED-AGAINST-HARDWARE — exact shape→byte mapping (e.g.
// `.128x256b` vs `.64x128b`) is hand-mapped; Phase 2 uses a single
// default 128-byte transfer (one TMEM slot) consistent with the
// existing `tcgen05.ld/st` placeholder behavior.

#include "ptxsim/instructions/tcgen05.h"

#include "ptx_ir/operand_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "utils/logger.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ptxsim {

[[noreturn]] void throw_cta_group_2(const char *instr_name) {
    PTX_ERROR_EMU("%s: .cta_group::2 is not supported "
                  "(cluster abstraction deferred to ADR-0018)",
                  instr_name);
    throw UnsupportedInstructionException(
        instr_name,
        std::string(instr_name) +
            ": .cta_group::2 is not yet supported (cluster abstraction "
            "deferred to ADR-0018, implement-cta-group-2-dist-smem)");
}

// Phase 2 placeholder: extract the smem offset from the source
// `AddrOperand`. Returns 0 if the operand is not an address with an
// immediate offset (e.g., register-offset or symbolic base). Future
// phases should resolve register offsets via the register bank.
uint32_t extract_smem_offset_placeholder(const Tcgen05Instr &instr) {
    if (instr.operands.size() < 2) {
        return 0;
    }
    const auto &op = instr.operands[1];
    if (op.kind() != OperandKind::ADDR) {
        return 0;
    }
    const auto &addr = std::get<AddrOperand>(op.data);
    if (addr.space != AddrOperand::Space::SHARED) {
        return 0; // not a shared-memory address → fall back to offset 0
    }
    if (addr.offsetType == AddrOperand::OffsetType::IMMEDIATE) {
        // Parse the textual immediate offset (e.g., "0x10", "32").
        try {
            long long parsed = std::stoll(addr.immediateOffset, nullptr, 0);
            if (parsed < 0)
                return 0;
            return static_cast<uint32_t>(parsed);
        } catch (...) {
            return 0;
        }
    }
    // REGISTER offset or symbolic base — placeholder, real
    // resolution deferred to a later phase that wires register-bank
    // lookups for tcgen05 operands.
    return 0;
}

void processTcgen05Cp(ThreadContext *context, const Tcgen05Instr &instr) {
    WarpContext *warp = context->get_warp_context();
    if (!warp) {
        PTX_ERROR_EMU("tcgen05.cp: no WarpContext attached to thread");
        throw UnsupportedInstructionException(
            "tcgen05.cp", "tcgen05.cp requires an active WarpContext");
    }
    CTAContext *cta = warp->get_cta_context();
    if (!cta) {
        PTX_ERROR_EMU("tcgen05.cp: no CTAContext attached to warp");
        throw UnsupportedInstructionException(
            "tcgen05.cp", "tcgen05.cp requires an active CTAContext");
    }

    // Oracle Q2-A: cta_group::2 not supported.
    if (instr.cta_group == 2) {
        throw_cta_group_2("tcgen05.cp");
    }

    // Oracle Q4-B: reuse `cta->sharedMemSpace` directly (Q4-B rejected
    // the `SmemDescriptor` abstraction).
    if (cta->sharedMemSpace == nullptr) {
        PTX_ERROR_EMU("tcgen05.cp: cta->sharedMemSpace is nullptr "
                      "(kernel declared no shared memory)");
        throw UnsupportedInstructionException(
            "tcgen05.cp",
            "tcgen05.cp requires a kernel with shared memory backing "
            "(cta->sharedMemSpace is null)");
    }

    const uint32_t smem_offset = extract_smem_offset_placeholder(instr);

    // Bounds check against the CTA's declared shared memory size.
    // `sharedMemBytes` is the user-requested size; cp accesses from
    // offset to offset + Tmem::kSlotSize.
    if (static_cast<uint64_t>(smem_offset) + Tmem::kSlotSize >
        cta->sharedMemBytes) {
        PTX_ERROR_EMU("tcgen05.cp: smem access [%u, +%zu) exceeds "
                      "sharedMemBytes=%zu",
                      smem_offset, Tmem::kSlotSize, cta->sharedMemBytes);
        throw std::runtime_error(
            "tcgen05.cp: shared memory access out of bounds");
    }

    // FU-3 C2 (Oracle Q5): allocate from per-warp cp cursor (offset by 32
    // to avoid overlap with ld/st slots 0..31). Real Blackwell cp has NO
    // slot operand in PTX; slot management is implicit.
    // Default cp slot: slot 32 + next_cp_slot_ (0, 1, 2, ...).
    constexpr size_t kCpBaseSlot = 32;
    size_t cp_slot = kCpBaseSlot + warp->allocate_cp_slot();

    uint8_t tmp[Tmem::kSlotSize];
    std::memcpy(tmp,
                static_cast<const uint8_t *>(cta->sharedMemSpace) + smem_offset,
                Tmem::kSlotSize);

    Tmem &tmem = cta->tmem();
    tmem.write(cp_slot, tmp, Tmem::kSlotSize);

    PTX_DEBUG_EMU("tcgen05.cp: smem[cta=%d +0x%x] (=%zu bytes) → "
                  "tmem slot %zu",
                  cta->blockIdx.x, smem_offset, Tmem::kSlotSize, cp_slot);
}

} // namespace ptxsim