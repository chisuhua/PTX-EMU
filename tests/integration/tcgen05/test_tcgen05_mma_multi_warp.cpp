// tests/integration/tcgen05/test_tcgen05_mma_multi_warp.cpp
// =============================================================================
// FU-4 of FlashAttention readiness: multi-warp fragment isolation tests
// (Oracle 2026-07-11 BLOCKER C4 fix).
//
// Verifies that tcgen05_fragment_mma_f16 with per-warp warp_id parameter
// writes C fragments to non-overlapping slot ranges [warp_id*32+64..warp_id*32+95].
//
// Backward compatibility: warp_id=0 path matches pre-C4 single-warp behavior.
// A/B slots [0..63] remain shared (per design D2 — minimal fix).
//
// UNVERIFIED-AGAINST-HARDWARE: fragment arithmetic is identical for all warps;
// the warp_id parameter controls only C slot isolation, not hardware semantics.
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/cta_context.h"
#include "ptxsim/instructions/tcgen05_helpers.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/utils/half_utils.h"

#include "reference/ptx_tcgen05/tcgen05_mma_golden.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

using ptxsim::reference::tcgen05::GOLDEN_MMA_F16_F16_F32;

namespace {

// Fill TMEM with the golden-input A and B fragments (same as other tcgen05 tests).
//
// Per tcgen05_helpers.cpp:
//   - lane `lane_id` reads A from slot `lane_id * 2` (128 bytes = 64 f16)
//   - lane `lane_id` reads B from slot `lane_id * 2 + 1` (first 32 f16 used)
//
// To produce the golden C, every lane must see the same A/B patterns.
void fill_tmem_with_golden_inputs(Tmem &tmem) {
    // A: 8 rows × 8 cols, only col 0 nonzero → write at f16 index i*8
    std::array<uint8_t, Tmem::kSlotSize> a_slot_buf{};
    for (int i = 0; i < 8; ++i) {
        const uint16_t h = f32_to_f16(static_cast<float>(i + 1));
        const size_t byte_idx = static_cast<size_t>(i) * 8 * 2;
        a_slot_buf[byte_idx]     = static_cast<uint8_t>(h & 0xFF);
        a_slot_buf[byte_idx + 1] = static_cast<uint8_t>(h >> 8);
    }
    // B: 8 rows × 4 cols, only row 0 nonzero → write at f16 index j
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

// Verify that C slots [base_slot .. base_slot+31] contain golden C values.
void verify_golden_c(Tmem &tmem, size_t base_slot) {
    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        size_t slot = base_slot + static_cast<size_t>(lane_id);
        std::array<uint8_t, Tmem::kSlotSize> buf{};
        tmem.read(slot, buf.data(), Tmem::kSlotSize);
        const float* c_frag = reinterpret_cast<const float*>(buf.data());

        for (int k = 0; k < 32; ++k) {
            REQUIRE(c_frag[k] == GOLDEN_MMA_F16_F16_F32[k]);
        }
    }
}

// Verify that C slots [base_slot .. base_slot+31] are zero (untouched).
void verify_zero_c(Tmem &tmem, size_t base_slot) {
    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        size_t slot = base_slot + static_cast<size_t>(lane_id);
        std::array<uint8_t, Tmem::kSlotSize> buf{};
        tmem.read(slot, buf.data(), Tmem::kSlotSize);
        const float* c_frag = reinterpret_cast<const float*>(buf.data());

        for (int k = 0; k < 32; ++k) {
            REQUIRE(c_frag[k] == 0.0f);
        }
    }
}

}  // namespace

// =============================================================================
// TC1: Single-warp backward compatibility
// =============================================================================
TEST_CASE("tcgen05_fragment_mma_f16 with warp_id=0 matches pre-C4 behavior",
          "[integration][tcgen05][mma][multi_warp][backward_compat]") {
    CTAContext cta;
    Tmem& tmem = cta.tmem();
    fill_tmem_with_golden_inputs(tmem);

    // Call with warp_id=0 (single-warp mode, backward compatible)
    ptxsim::tcgen05_fragment_mma_f16(tmem, /*warp_id=*/0, /*accumulate=*/false);

    // C slots should be at [64..95] (same as pre-C4 behavior)
    verify_golden_c(tmem, /*base_slot=*/64);
}

// =============================================================================
// TC2: warp 0 writes C to slots [64..95]
// =============================================================================
TEST_CASE("tcgen05_fragment_mma_f16 warp 0 writes C to slots [64..95]",
          "[integration][tcgen05][mma][multi_warp][warp0]") {
    CTAContext cta;
    Tmem& tmem = cta.tmem();
    fill_tmem_with_golden_inputs(tmem);

    ptxsim::tcgen05_fragment_mma_f16(tmem, /*warp_id=*/0, /*accumulate=*/false);

    // warp 0 owns [64..95]
    verify_golden_c(tmem, /*base_slot=*/64);
}

// =============================================================================
// TC3: warp 1 writes C to slots [96..127] (NOT [64..95]!)
// =============================================================================
TEST_CASE("tcgen05_fragment_mma_f16 warp 1 writes C to slots [96..127]",
          "[integration][tcgen05][mma][multi_warp][warp1]") {
    CTAContext cta;
    Tmem& tmem = cta.tmem();
    fill_tmem_with_golden_inputs(tmem);

    ptxsim::tcgen05_fragment_mma_f16(tmem, /*warp_id=*/1, /*accumulate=*/false);

    // warp 1 owns [96..127], NOT [64..95]
    verify_golden_c(tmem, /*base_slot=*/96);
    // warp 0's range should be untouched
    verify_zero_c(tmem, /*base_slot=*/64);
}

// =============================================================================
// TC4: 2-warps in parallel do not conflict on C slots
// =============================================================================
TEST_CASE("2 warps mma in parallel do not conflict on C slot",
          "[integration][tcgen05][mma][multi_warp][no_conflict]") {
    CTAContext cta;
    Tmem& tmem = cta.tmem();
    fill_tmem_with_golden_inputs(tmem);

    // Run warp 0 mma
    ptxsim::tcgen05_fragment_mma_f16(tmem, /*warp_id=*/0, /*accumulate=*/false);
    // Run warp 1 mma
    ptxsim::tcgen05_fragment_mma_f16(tmem, /*warp_id=*/1, /*accumulate=*/false);

    // Both warps' C ranges should have golden values
    verify_golden_c(tmem, /*base_slot=*/64);   // warp 0
    verify_golden_c(tmem, /*base_slot=*/96);   // warp 1
}

// =============================================================================
// TC5: warp_id negative throws std::invalid_argument
// =============================================================================
TEST_CASE("tcgen05_fragment_mma_f16 with warp_id=-1 throws invalid_argument",
          "[integration][tcgen05][mma][multi_warp][exception]") {
    CTAContext cta;
    Tmem& tmem = cta.tmem();
    fill_tmem_with_golden_inputs(tmem);

    REQUIRE_THROWS_AS(
        ptxsim::tcgen05_fragment_mma_f16(tmem, /*warp_id=*/-1, /*accumulate=*/false),
        std::invalid_argument);
}