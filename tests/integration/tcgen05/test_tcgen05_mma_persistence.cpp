// tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp
// =============================================================================
// Phase A Step 3 of Oracle 2026-07-10 FlashAttention-readiness audit:
// multi-op TMEM persistence test (validates Oracle H5 hypothesis).
//
// CONTEXT:
//   Oracle 2026-07-10 reported (H5, confidence: HIGH):
//     "无多操作 TMEM 持久化测试（mma → 中间操作 → mma）"
//   This file adds the missing chained-op coverage. Prior to this test:
//     - integration_tcgen05_mma_ws (3 TC) — single mma only
//     - integration_tcgen05_cp (3 TC) — single cp only
//     No test exercised `mma → cp → mma` or repeated mma on the same TMEM.
//
// WHAT THIS VERIFIES:
//   T1. Two consecutive processTcgen05Mma calls with identical A,B →
//       observe whether the helper ACCUMULATES into C slot or OVERWRITES.
//       This is the H1 (accumulator) test. Oracle H1 prediction: overwrite.
//   T2. processTcgen05Cp after processTcgen05Mma preserves C output —
//       cp writes slot 0, mma wrote slot[64..95]; cross-slot isolation.
//   T3. Full FlashAttention-style chain mma → cp → mma — the 2nd mma
//       consumes cp-loaded A/B; C slot gets re-written by the 2nd mma.
//
// UNVERIFIED-AGAINST-HARDWARE — golden values are hand-computed per
// tests/reference/ptx_tcgen05/tcgen05_mma_golden.h (PTX ISA §9.7.16).
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instructions/tcgen05.h"
#include "ptxsim/instructions/tcgen05_helpers.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/half_utils.h"
#include "ptxsim/warp_context.h"

#include "reference/ptx_tcgen05/tcgen05_mma_golden.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

using ptxsim::reference::tcgen05::GOLDEN_MMA_F16_F16_F32;

namespace {

// Minimal TestRig mirroring tests/integration/tcgen05/test_tcgen05_mma_ws.cpp
// but with shared-memory backing for cp testing.
class TestRig {
public:
    explicit TestRig(size_t smem_bytes = 4096)
        : sm_(std::make_unique<SMContext>(/*num_warps=*/1, /*warp_size=*/32,
                                          /*max_ctas=*/1,
                                          /*shared_mem=*/4096)),
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

// Build a minimal Tcgen05Instr for regular (non-ws) mma.
// Empty qualifiers + 4 placeholder operands is enough for the regular path
// (ws path would require Q_F16 + Q_TCGEN_WS to pass the Q3-A scope check).
Tcgen05Instr make_regular_mma_instr() {
    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::MMA;
    instr.cta_group = 1;
    instr.dtype = Tcgen05Dtype::F16;
    instr.operands = std::vector<OperandContext>(
        4, OperandContext(RegOperand{"r", 0}));
    return instr;
}

// Build a Tcgen05Instr for tcgen05.cp with the given smem offset.
// Mirrors tests/integration/tcgen05/test_tcgen05_cp.cpp:64 helper.
Tcgen05Instr make_cp_instr_with_offset(uint32_t smem_offset) {
    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::CP;
    instr.cta_group = 1;

    AddrOperand dst_dummy;
    dst_dummy.space = AddrOperand::Space::SHARED;
    dst_dummy.offsetType = AddrOperand::OffsetType::IMMEDIATE;
    dst_dummy.immediateOffset = "0";
    instr.operands.push_back(OperandContext(dst_dummy));

    AddrOperand src;
    src.space = AddrOperand::Space::SHARED;
    src.offsetType = AddrOperand::OffsetType::IMMEDIATE;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%u", smem_offset);
    src.immediateOffset = buf;
    instr.operands.push_back(OperandContext(src));

    return instr;
}

// Fill A/B TMEM slots with golden inputs (per tcgen05_mma_golden.h).
// Per-lane layout (see tcgen05_helpers.h:20-22):
//   - a_slot = lane_id * 2
//   - b_slot = lane_id * 2 + 1
void fill_tmem_with_golden_inputs(Tmem &tmem) {
    std::array<uint8_t, Tmem::kSlotSize> a_slot_buf{};
    for (int i = 0; i < 8; ++i) {
        const uint16_t h = f32_to_f16(static_cast<float>(i + 1));
        const size_t byte_idx = static_cast<size_t>(i) * 8 * 2;
        a_slot_buf[byte_idx]     = static_cast<uint8_t>(h & 0xFF);
        a_slot_buf[byte_idx + 1] = static_cast<uint8_t>(h >> 8);
    }

    std::array<uint8_t, Tmem::kSlotSize> b_slot_buf{};
    for (int j = 0; j < 4; ++j) {
        const uint16_t h = f32_to_f16(static_cast<float>(j + 1));
        b_slot_buf[j * 2]     = static_cast<uint8_t>(h & 0xFF);
        b_slot_buf[j * 2 + 1] = static_cast<uint8_t>(h >> 8);
    }

    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        tmem.write(static_cast<size_t>(lane_id) * 2, a_slot_buf.data(),
                   Tmem::kSlotSize);
        tmem.write(static_cast<size_t>(lane_id) * 2 + 1, b_slot_buf.data(),
                   Tmem::kSlotSize);
    }
}

// Read C fragments from slot[64..95] and compare against `expected`
// (length-32 f32 array). TMEM stores f16; we re-convert for comparison.
void require_c_slot_matches(Tmem &tmem,
                            const std::array<float, 32> &expected,
                            const char *context_info) {
    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        std::array<uint8_t, Tmem::kSlotSize> c_buf{};
        tmem.read(static_cast<size_t>(64) + static_cast<size_t>(lane_id),
                  c_buf.data(), Tmem::kSlotSize);

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 4; ++j) {
                const int idx = i * 4 + j;
                const uint16_t actual_bits = static_cast<uint16_t>(
                    c_buf[idx * 2] | (c_buf[idx * 2 + 1] << 8));
                const float actual = f16_to_f32(actual_bits);
                INFO(context_info << " lane=" << lane_id << " i=" << i
                     << " j=" << j << " expected=" << expected[idx]
                     << " actual=" << actual);
                REQUIRE(actual == Catch::Approx(expected[idx]));
            }
        }
    }
}

} // namespace

// =============================================================================
// T1: Repeated mma → observe overwrite-vs-accumulate behavior (H1 oracle).
// =============================================================================

TEST_CASE("processTcgen05Mma called twice with identical A,B leaves C "
          "unchanged (overwrite, not accumulate)",
          "[integration][tcgen05][mma][persistence][overwrite]") {
    TestRig rig;
    fill_tmem_with_golden_inputs(rig.tmem());

    auto instr = make_regular_mma_instr();

    // 1st mma — should produce golden C.
    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), instr));
    require_c_slot_matches(rig.tmem(), GOLDEN_MMA_F16_F16_F32,
                           "after 1st mma");

    // 2nd mma on the same inputs. Oracle H1 prediction: overwrite (no
    // accumulator). If helper accidentally accumulated, golden values
    // would become 2*(i+1)*(j+1); this assertion would then fail.
    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), instr));
    require_c_slot_matches(rig.tmem(), GOLDEN_MMA_F16_F16_F32,
                           "after 2nd mma (same inputs)");
}

// =============================================================================
// T2: mma → cp preserves C output (cross-slot isolation).
// =============================================================================

TEST_CASE("processTcgen05Cp after processTcgen05Mma preserves C output "
          "(cp writes slot 0, mma wrote slot 64..95 — no cross-slot "
          "interference)",
          "[integration][tcgen05][mma][cp][persistence][slot_isolation]") {
    TestRig rig;
    fill_tmem_with_golden_inputs(rig.tmem());

    auto mma = make_regular_mma_instr();
    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), mma));

    // cp from smem[256] → TMEM slot 0 (128 bytes). This should NOT touch
    // slot[64..95] where the C output from the mma resides.
    constexpr uint32_t kSmemOffset = 256;
    for (size_t i = 0; i < Tmem::kSlotSize; ++i) {
        rig.smem()[kSmemOffset + i] =
            static_cast<uint8_t>(0xC0 + (i & 0x0F));
    }

    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Cp(&rig.thread(),
                                 make_cp_instr_with_offset(kSmemOffset)));

    // C output must survive cp.
    require_c_slot_matches(rig.tmem(), GOLDEN_MMA_F16_F16_F32,
                           "after mma -> cp (C should be preserved)");

    // Slot 0 should hold the cp'd byte pattern (smem[256] byte stream).
    uint8_t slot0[Tmem::kSlotSize];
    rig.tmem().read(0, slot0, Tmem::kSlotSize);
    for (size_t i = 0; i < Tmem::kSlotSize; ++i) {
        INFO("cp->slot0 byte " << i << ": expected="
             << static_cast<int>(0xC0 + (i & 0x0F))
             << " actual=" << static_cast<int>(slot0[i]));
        REQUIRE(slot0[i] == static_cast<uint8_t>(0xC0 + (i & 0x0F)));
    }
}

// =============================================================================
// T3: mma → cp → mma pipeline (FlashAttention-style chained ops).
// =============================================================================

TEST_CASE("processTcgen05Mma -> processTcgen05Cp -> processTcgen05Mma "
          "chain runs without throw and 2nd mma observes cp-loaded data",
          "[integration][tcgen05][mma][cp][persistence][chain]") {
    TestRig rig;
    fill_tmem_with_golden_inputs(rig.tmem());

    auto mma = make_regular_mma_instr();

    // 1st mma — produces golden C in slot[64..95].
    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), mma));
    require_c_slot_matches(rig.tmem(), GOLDEN_MMA_F16_F16_F32,
                           "1st mma (golden C)");

    // cp overwrites slot[0] from smem (128 bytes of pattern 0xAA).
    constexpr uint32_t kSmemOffset = 0;
    for (size_t i = 0; i < Tmem::kSlotSize; ++i) {
        rig.smem()[kSmemOffset + i] = 0xAA;
    }
    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Cp(&rig.thread(),
                                 make_cp_instr_with_offset(kSmemOffset)));

    // 2nd mma — helper reads lane 0's A from slot 0 (now 0xAA pattern)
    // and lane 0's B from slot 1 (still original B = {1..4}). So lane 0's
    // C is 0xAA * b_k_j (mix). We only verify the call does NOT throw and
    // that C changes for at least some lane (proving cp data was consumed
    // and helper observed new state — a stronger check than "no throw").
    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), mma));

    bool at_least_one_element_changed = false;
    for (int lane_id = 0; lane_id < 32 && !at_least_one_element_changed;
         ++lane_id) {
        std::array<uint8_t, Tmem::kSlotSize> c_buf{};
        rig.tmem().read(static_cast<size_t>(64) +
                            static_cast<size_t>(lane_id),
                        c_buf.data(), Tmem::kSlotSize);
        for (int idx = 0; idx < 32 && !at_least_one_element_changed; ++idx) {
            const float actual = f16_to_f32(static_cast<uint16_t>(
                c_buf[idx * 2] | (c_buf[idx * 2 + 1] << 8)));
            if (actual != Catch::Approx(GOLDEN_MMA_F16_F16_F32[idx])) {
                at_least_one_element_changed = true;
            }
        }
    }
    REQUIRE(at_least_one_element_changed);
}
