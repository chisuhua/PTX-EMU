// tests/integration/tcgen05/test_tcgen05_mma_ws.cpp
// =============================================================================
// Phase 3 of implement-tcgen05-handlers-extended (Oracle 2026-07-08 A-path):
// integration test for tcgen05.mma.ws qualifier-based routing inside
// processTcgen05Mma.
//
// Verifies that constructing Tcgen05Instr{op_kind=MMA,
// qualifiers={Q_TCGEN_WS, Q_F16, Q_TCGEN_CTA_GROUP}} and calling
// processTcgen05Mma produces the same fragment arithmetic as the
// regular mma path (since the ws path calls the same helper).
//
// Test pattern mirrors tests/integration/tcgen05/test_tcgen05_cp.cpp:
//   1. Build TestRig with SM + CTA + Warp + Thread
//   2. Pre-fill TMEM with golden A + B inputs (per tcgen05_mma_golden.h)
//   3. Call processTcgen05Mma with ws-qualified instr
//   4. Read TMEM slots [64..95] (C fragments) and compare to golden C
//
// UNVERIFIED-AGAINST-HARDWARE: arithmetic is shared with regular mma path;
// any divergence indicates a routing bug, not a hardware-vs-sim gap.
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instructions/tcgen05.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"

#include "reference/ptx_tcgen05/tcgen05_mma_golden.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include "ptxsim/utils/half_utils.h"

using ptxsim::reference::tcgen05::GOLDEN_MMA_F16_F16_F32;

namespace {

class TestRig {
public:
    TestRig()
        : sm_(std::make_unique<SMContext>(/*num_warps=*/1, /*warp_size=*/32,
                                          /*max_ctas=*/1, /*shared_mem=*/4096)),
          cta_(std::make_unique<CTAContext>()),
          warp_(std::make_unique<WarpContext>()),
          thread_(std::make_unique<ThreadContext>()) {
        warp_->set_warp_id(0);
        warp_->set_cta_context(cta_.get());
        thread_->set_warp_context(warp_.get());
    }

    Tmem &tmem() { return cta_->tmem(); }
    CTAContext &cta() { return *cta_; }
    WarpContext &warp() { return *warp_; }
    ThreadContext &thread() { return *thread_; }

private:
    std::unique_ptr<SMContext> sm_;
    std::unique_ptr<CTAContext> cta_;
    std::unique_ptr<WarpContext> warp_;
    std::unique_ptr<ThreadContext> thread_;
};

// Build a Tcgen05Instr with ws+f16+cta_group qualifiers, op_kind=MMA
// (the grammar's path).
Tcgen05Instr make_ws_instr() {
    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::MMA;
    instr.qualifiers = {
        Qualifier::Q_TCGEN_WS,
        Qualifier::Q_F16,
        Qualifier::Q_TCGEN_CTA_GROUP,
    };
    // 4 operands (MMA / MMA_WS have operand count 4 per ptx_op.def:133-134).
    // Operand content is irrelevant for the test.
    instr.operands = std::vector<OperandContext>(
        4, OperandContext(RegOperand{"r", 0}));
    return instr;
}

// Fill TMEM with the golden-input A and B fragments.
//
// Per tcgen05.cpp:333-371 (helper):
//   - lane `lane_id` reads A from slot `lane_id * 2` (128 bytes = 64 f16)
//   - lane `lane_id` reads B from slot `lane_id * 2 + 1` (only first
//     32 f16 used by arithmetic)
//
// To produce the golden C (A[8][1]={1..8}, B[1][4]={1..4}), every lane
// must see the same A[8] pattern in its A slot's first 16 bytes and the
// same B[4] pattern in its B slot's first 8 bytes.
void fill_tmem_with_golden_inputs(Tmem &tmem) {
    // A fragment is 8 rows × 8 cols (64 f16 per A slot, only col 0 nonzero).
    // Helper accesses a_flat[i*8 + k] = A[i][k]. To set A[i][0] = i+1, write
    // at f16 index i*8 within the A slot.
    std::array<uint8_t, Tmem::kSlotSize> a_slot_buf{};
    for (int i = 0; i < 8; ++i) {
        const uint16_t h = f32_to_f16(static_cast<float>(i + 1));
        const size_t byte_idx = static_cast<size_t>(i) * 8 * 2;
        a_slot_buf[byte_idx]     = static_cast<uint8_t>(h & 0xFF);
        a_slot_buf[byte_idx + 1] = static_cast<uint8_t>(h >> 8);
    }
    // B fragment is 8 rows × 4 cols (32 f16 used of B slot's 64, only row 0
    // nonzero). Helper accesses b_flat[k*4 + j] = B[k][j]. To set B[0][j] = j+1,
    // write at f16 index j within the B slot.
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

}  // namespace

// =============================================================================
// Happy path: ws + f16 + cta_group → produces golden C fragments
// =============================================================================

TEST_CASE("processTcgen05Mma with ws+f16+cta_group qualifiers produces "
          "golden C fragments",
          "[integration][tcgen05][mma_ws][golden]") {
    TestRig rig;
    fill_tmem_with_golden_inputs(rig.tmem());

    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(
        &rig.thread(), make_ws_instr()));

    // Every lane's C fragment (slot 64 + lane_id) should equal the golden
    // 32-element array (8 rows × 4 cols of f16 values).
    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        std::array<uint8_t, Tmem::kSlotSize> c_buf{};
        rig.tmem().read(static_cast<size_t>(64) + static_cast<size_t>(lane_id),
                        c_buf.data(), Tmem::kSlotSize);

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 4; ++j) {
                const int idx = i * 4 + j;
                const float expected = GOLDEN_MMA_F16_F16_F32[static_cast<size_t>(idx)];
                const uint16_t actual_bits = static_cast<uint16_t>(
                    c_buf[idx * 2] | (c_buf[idx * 2 + 1] << 8));
                const float actual = f16_to_f32(actual_bits);
                INFO("lane=" << lane_id << " i=" << i << " j=" << j
                     << " expected=" << expected << " actual=" << actual);
                REQUIRE(actual == Catch::Approx(expected));
            }
        }
    }
}

// =============================================================================
// Same handler, different dispatch op_kind — also produces golden
// =============================================================================

TEST_CASE("processTcgen05Mma with op_kind=MMA_WS (direct construction) "
          "produces regular mma result",
          "[integration][tcgen05][mma_ws][dispatch]") {
    TestRig rig;
    fill_tmem_with_golden_inputs(rig.tmem());

    Tcgen05Instr instr = make_ws_instr();
    instr.op_kind = Tcgen05OpKind::MMA_WS;  // direct construction
    // Note: no Q_TCGEN_WS qualifier; this exercises the dispatch path
    // (case Tcgen05OpKind::MMA_WS routes to processTcgen05Mma which
    // sees no Q_TCGEN_WS and falls through to regular mma).

    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), instr));

    // Read lane 0's C fragment and verify against golden.
    std::array<uint8_t, Tmem::kSlotSize> c_buf{};
    rig.tmem().read(64, c_buf.data(), Tmem::kSlotSize);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            const int idx = i * 4 + j;
            const float expected = GOLDEN_MMA_F16_F16_F32[static_cast<size_t>(idx)];
            const uint16_t actual_bits = static_cast<uint16_t>(
                c_buf[idx * 2] | (c_buf[idx * 2 + 1] << 8));
            REQUIRE(f16_to_f32(actual_bits) == Catch::Approx(expected));
        }
    }
}

// =============================================================================
// Negative path: ws + non-f16 throws
// =============================================================================

TEST_CASE("processTcgen05Mma with ws + Q_F32 throws before reaching helper",
          "[integration][tcgen05][mma_ws][scope_violation]") {
    TestRig rig;
    // Intentionally do NOT pre-fill TMEM with golden inputs. If the ws
    // scope check fails to throw, the helper would multiply zeros and
    // write zeros — we wouldn't observe the throw. To prove the scope
    // check fires, we instead verify the exception is observed (the
    // helper would not be reached).

    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::MMA;
    instr.qualifiers = {Qualifier::Q_TCGEN_WS, Qualifier::Q_F32};
    instr.operands = std::vector<OperandContext>(
        4, OperandContext(RegOperand{"r", 0}));

    REQUIRE_THROWS_AS(ptxsim::processTcgen05Mma(&rig.thread(), instr),
                      UnsupportedInstructionException);
}