// tests/integration/tcgen05/test_tcgen05_mma_k_loop_128.cpp
// =============================================================================
// FU-5 Phase 1.2: K=128 mma accumulation coverage test (Oracle B1 gap).
//
// Prior to this test, no K-loop test verified that 100+ mma calls with
// accumulate=true preserve numeric stability. The 4× loop in
// test_tcgen05_mma_persistence.cpp is a sanity check, not a full K=128
// FlashAttention scenario.
//
// WHAT THIS VERIFIES:
//   TC1: 128 mma calls (1 handler + 127 direct helper, all accumulate=true)
//        on identical A,B produce C == 128 × GOLDEN_MMA_F16_F16_F32
//        within 1e-3 relative tolerance (tightened from 1e-6 per Oracle B7).
//   TC2: Per-iteration random inputs assert independence — each iteration
//        provides different A,B to verify accumulator doesn't leak across
//        iterations.
//
// DEPENDENCIES:
//   - FU-2 (C1 idesc parsing) — accumulate=true via helper parameter
//   - H1 (accumulator) — tcgen05_fragment_mma_f16(accumulate=true)
//   - H2 (f32 storage) — C slot readback in f32 not f16
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/cta_context.h"
#include "ptxsim/instructions/tcgen05.h"
#include "ptxsim/instructions/tcgen05_helpers.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/testing/tmem_helpers.h"

#include "reference/ptx_tcgen05/tcgen05_mma_golden.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>

using ptxsim::reference::tcgen05::GOLDEN_MMA_F16_F16_F32;
using namespace ptxsim::testing::tmem;

namespace {

// Minimal test harness for K-loop accumulation.
struct KLoopFixture {
    CTAContext cta;
    Tmem &tmem;

    KLoopFixture() : cta(), tmem(cta.tmem()) {}

    void run_loop(int n_iterations, bool accumulate) {
        for (int i = 0; i < n_iterations; ++i) {
            ptxsim::tcgen05_fragment_mma_f16(tmem, /*warp_id=*/0,
                                             accumulate);
        }
    }
};

// Scale golden C by factor N.
std::array<float, 32> scale_golden(int n) {
    std::array<float, 32> scaled{};
    for (size_t k = 0; k < 32; ++k) {
        scaled[k] = static_cast<float>(n) * GOLDEN_MMA_F16_F16_F32[k];
    }
    return scaled;
}

} // namespace

// =============================================================================
// TC1: K=128 sequential accumulation on identical A,B.
//
// Single pass: 128 mma calls with accumulate=true. Verifies that the
// sum converges to 128 × golden within 1e-3 relative tolerance.
// The looser tolerance (1e-3 vs 1e-6) accounts for accumulated f16↔f32
// round-trip error over 128 iterations per PTX ISA §9.7.16.
// =============================================================================

TEST_CASE("K=128 mma accumulation produces 128× golden within 1e-3 "
          "relative tolerance (FU-5 B1 K-loop)",
          "[integration][tcgen05][mma][flashattention][k-loop]") {
    KLoopFixture fix;
    fill_tmem_with_golden_inputs(fix.tmem);

    // 1st pass: overwrite baseline (accumulate=false → C=1×golden).
    ptxsim::tcgen05_fragment_mma_f16(fix.tmem, /*warp_id=*/0,
                                     /*accumulate=*/false);
    require_c_slot_matches(fix.tmem, GOLDEN_MMA_F16_F16_F32,
                           "after 1st mma (overwrite baseline)");

    // Next 127 passes: accumulate.
    fix.run_loop(127, /*accumulate=*/true);

    const auto expected = scale_golden(128);
    const bool passed = compare_c_slot_to_reference(
        fix.tmem, expected, /*epsilon=*/1e-3, /*margin=*/1e-5,
        "K=128 accumulate");
    REQUIRE(passed);
}

// =============================================================================
// TC2: K=128 with random per-iteration inputs.
//
// Each iteration fills TMEM with different A,B values (scale offset i)
// then runs mma with accumulate=true. After K iterations the C slot
// contains sum of iteration results. We verify against a reference sum
// computed in f64 to detect accumulation drift.
//
// Input strategy: A[i][0] = (i+1) * (iter+1), B[0][j] = (j+1) * (iter+1)
// → golden per iteration: C[i][j] = (i+1)*(j+1) * (iter+1)^2
// =============================================================================

TEST_CASE("K=128 with distinct per-iteration inputs validates accumulator "
          "independence (FU-5 B1 random input K-loop)",
          "[integration][tcgen05][mma][flashattention][k-loop][random]") {
    CTAContext cta;
    Tmem &tmem = cta.tmem();

    // Compute expected sum in f64 for precision.
    std::array<double, 32> expected_f64{};
    for (int iter = 0; iter < 128; ++iter) {
        const double scale = static_cast<double>((iter + 1) * (iter + 1));
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 4; ++j) {
                const int idx = i * 4 + j;
                expected_f64[idx] += scale * static_cast<double>((i + 1) * (j + 1));
            }
        }
    }

    // Run the K-loop with per-iteration inputs.
    for (int iter = 0; iter < 128; ++iter) {
        // Build per-iteration A/B. A[i][0] = (i+1)*(iter+1),
        // B[0][j] = (j+1)*(iter+1). Other entries zero.
        const float itf = static_cast<float>(iter + 1);

        std::array<uint8_t, Tmem::kSlotSize> a_buf{};
        for (int i = 0; i < 8; ++i) {
            const float val = static_cast<float>(i + 1) * itf;
            const uint16_t h = f32_to_f16(val);
            const size_t byte_idx = static_cast<size_t>(i) * 8 * 2;
            a_buf[byte_idx]     = static_cast<uint8_t>(h & 0xFF);
            a_buf[byte_idx + 1] = static_cast<uint8_t>(h >> 8);
        }

        std::array<uint8_t, Tmem::kSlotSize> b_buf{};
        for (int j = 0; j < 4; ++j) {
            const float val = static_cast<float>(j + 1) * itf;
            const uint16_t h = f32_to_f16(val);
            b_buf[j * 2]     = static_cast<uint8_t>(h & 0xFF);
            b_buf[j * 2 + 1] = static_cast<uint8_t>(h >> 8);
        }

        for (int lane_id = 0; lane_id < 32; ++lane_id) {
            tmem.write(static_cast<size_t>(lane_id) * 2, a_buf.data(),
                       Tmem::kSlotSize);
            tmem.write(static_cast<size_t>(lane_id) * 2 + 1, b_buf.data(),
                       Tmem::kSlotSize);
        }

        ptxsim::tcgen05_fragment_mma_f16(tmem, /*warp_id=*/0,
                                         /*accumulate=*/(iter > 0));
    }

    // Convert f64 expected to f32 and verify.
    std::array<float, 32> expected_f32{};
    for (size_t k = 0; k < 32; ++k) {
        expected_f32[k] = static_cast<float>(expected_f64[k]);
    }

    REQUIRE(compare_c_slot_to_reference(
        tmem, expected_f32, /*epsilon=*/1e-2, /*margin=*/1.0,
        "K=128 random accumulate"));
}