// tests/integration/tcgen05/test_tcgen05_mma_cp_data_flow.cpp
// =============================================================================
// FU-5 Phase 3: cp → mma data flow integration test (Oracle B4 gap).
//
// Corrected design (from original flawed concept):
//   cp writes to per-warp cursor slot (slot 32 for warp 0, per FU-3 C2),
//   NOT to mma A/B slots [0..63]. This test verifies:
//
//   TC1: mma produces golden C → cp writes data to cp slot 32 →
//        C output is preserved (cross-slot isolation, quantitative).
//   TC2: cp loads distinct patterns to multiple offsets, verified via
//        direct TMEM read of consecutive cp slots.
//
// DEPENDENCIES:
//   - FU-3 (C2 ld/st slot routing) — cp writes to per-warp cursor slot 32
//   - H1 (accumulator) + H2 (f32 storage)
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instructions/tcgen05.h"
#include "ptxsim/instructions/tcgen05_helpers.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/testing/tmem_helpers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"

#include "reference/ptx_tcgen05/tcgen05_mma_golden.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

using ptxsim::reference::tcgen05::GOLDEN_MMA_F16_F16_F32;
using namespace ptxsim::testing::tmem;

namespace {

class TestRig {
public:
    explicit TestRig(size_t smem_bytes = 4096)
        : sm_(std::make_unique<SMContext>(1, 32, 1, 4096)),
          cta_(std::make_unique<CTAContext>()),
          warp_(std::make_unique<WarpContext>()),
          thread_(std::make_unique<ThreadContext>()),
          smem_buf_(smem_bytes, 0) {
        warp_->set_warp_id(0);
        warp_->set_cta_context(cta_.get());
        thread_->set_warp_context(warp_.get());
        cta_->sharedMemBytes = smem_bytes;
        cta_->sharedMemSpace = smem_buf_.data();
    }

    CTAContext &cta() { return *cta_; }
    WarpContext &warp() { return *warp_; }
    ThreadContext &thread() { return *thread_; }
    Tmem &tmem() { return cta_->tmem(); }
    std::vector<uint8_t> &smem() { return smem_buf_; }

private:
    std::unique_ptr<SMContext> sm_;
    std::unique_ptr<CTAContext> cta_;
    std::unique_ptr<WarpContext> warp_;
    std::unique_ptr<ThreadContext> thread_;
    std::vector<uint8_t> smem_buf_;
};

ptxemu::ir::Tcgen05Instr make_mma_instr() {
    ptxemu::ir::Tcgen05Instr instr;
    instr.op_kind = ptxemu::ir::Tcgen05OpKind::MMA;
    instr.cta_group = 1;
    instr.dtype = ptxemu::ir::Tcgen05Dtype::F16;
    instr.operands = std::vector<ptxemu::ir::OperandContext>(
        4, ptxemu::ir::OperandContext(RegOperand{"r", 0}));
    return instr;
}

ptxemu::ir::Tcgen05Instr make_cp_instr(uint32_t smem_offset) {
    ptxemu::ir::Tcgen05Instr instr;
    instr.op_kind = ptxemu::ir::Tcgen05OpKind::CP;
    instr.cta_group = 1;
    AddrOperand dst;
    dst.space = AddrOperand::Space::SHARED;
    dst.offsetType = AddrOperand::OffsetType::IMMEDIATE;
    dst.immediateOffset = "0";
    instr.operands.push_back(ptxemu::ir::OperandContext(dst));
    AddrOperand src;
    src.space = AddrOperand::Space::SHARED;
    src.offsetType = AddrOperand::OffsetType::IMMEDIATE;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%u", smem_offset);
    src.immediateOffset = buf;
    instr.operands.push_back(ptxemu::ir::OperandContext(src));
    return instr;
}

} // namespace

// =============================================================================
// TC1: mma → cp preserves C output with quantitative assertions.
//
// Strategy:
//   1. Fill A/B with golden inputs, run mma → verify golden C
//   2. Write distinctive pattern (0xCC) to smem at offset 256
//   3. cp loads from smem[256] → TMEM cp slot (FU-3 C2: slot 32)
//   4. Verify C output is still golden (cross-slot isolation)
//   5. Verify cp data is actually at slot 32
//
// This is stronger than persistence T3: quantitative verification
// of C preservation (not just "at least one element changed").
// =============================================================================

TEST_CASE("mma produces golden C → cp writes to cp slot → C preserved "
          "(FU-5 B4 cross-slot isolation, quantitative)",
          "[integration][tcgen05][cp][mma][flashattention][data-flow]") {
    TestRig rig;

    // Step 1: Run mma on golden inputs.
    fill_tmem_with_golden_inputs(rig.tmem());
    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), make_mma_instr()));
    require_c_slot_matches(rig.tmem(), GOLDEN_MMA_F16_F16_F32,
                           "after 1st mma (golden C)");

    // Step 2: Write distinctive pattern (0xCC) to smem.
    constexpr uint32_t kSmemOffset = 256;
    for (size_t i = 0; i < Tmem::kSlotSize; ++i) {
        rig.smem()[kSmemOffset + i] = static_cast<uint8_t>(0xCC);
    }

    // Step 3: cp from smem[256] → TMEM cp slot.
    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Cp(&rig.thread(),
                                 make_cp_instr(kSmemOffset)));

    // Step 4: C output must survive cp (quantitative).
    require_c_slot_matches(rig.tmem(), GOLDEN_MMA_F16_F16_F32,
                           "after cp (C should survive)");

    // Step 5: cp data at slot 32 (FU-3 C2: base=32, cursor=0).
    uint8_t cp_slot[Tmem::kSlotSize];
    rig.tmem().read(32, cp_slot, Tmem::kSlotSize);
    for (size_t i = 0; i < Tmem::kSlotSize; ++i) {
        INFO("cp slot 32 byte " << i << ": expected=0xCC actual="
             << static_cast<int>(cp_slot[i]));
        REQUIRE(cp_slot[i] == static_cast<uint8_t>(0xCC));
    }
}

// =============================================================================
// TC2: cp loads distinct patterns to multiple offsets, verified via
//      direct TMEM read of consecutive cp slots.
//
// Per FU-3 C2: cp uses per-warp cursor starting at slot 32. Two
// consecutive cp calls should write to slots 32 and 33.
// =============================================================================

TEST_CASE("cp loads 2 distinct patterns → slots 32 and 33 hold correct data "
          "(FU-5 B4 per-slot cp integrity)",
          "[integration][tcgen05][cp][flashattention][data-flow][per-slot]") {
    TestRig rig;

    // Write two distinct patterns to smem at different offsets.
    constexpr uint32_t kOff1 = 0;
    constexpr uint32_t kOff2 = 128;
    for (size_t i = 0; i < Tmem::kSlotSize; ++i) {
        rig.smem()[kOff1 + i] = static_cast<uint8_t>(0xAA);
        rig.smem()[kOff2 + i] = static_cast<uint8_t>(0xBB);
    }

    // First cp: smem[0] → slot 32.
    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Cp(&rig.thread(), make_cp_instr(kOff1)));

    uint8_t slot32[Tmem::kSlotSize];
    rig.tmem().read(32, slot32, Tmem::kSlotSize);
    for (size_t i = 0; i < Tmem::kSlotSize; ++i) {
        INFO("slot32[" << i << "]");
        REQUIRE(slot32[i] == static_cast<uint8_t>(0xAA));
    }

    // Second cp: smem[128] → slot 33 (cursor incremented).
    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Cp(&rig.thread(), make_cp_instr(kOff2)));

    uint8_t slot33[Tmem::kSlotSize];
    rig.tmem().read(33, slot33, Tmem::kSlotSize);
    for (size_t i = 0; i < Tmem::kSlotSize; ++i) {
        INFO("slot33[" << i << "]");
        REQUIRE(slot33[i] == static_cast<uint8_t>(0xBB));
    }

    // Slot 32 should still have 0xAA (not overwritten).
    uint8_t slot32_again[Tmem::kSlotSize];
    rig.tmem().read(32, slot32_again, Tmem::kSlotSize);
    for (size_t i = 0; i < Tmem::kSlotSize; ++i) {
        INFO("slot32_again[" << i << "]");
        REQUIRE(slot32_again[i] == static_cast<uint8_t>(0xAA));
    }
}