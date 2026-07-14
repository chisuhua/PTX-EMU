#ifndef PTXSIM_TESTING_TMEM_HELPERS_H
#define PTXSIM_TESTING_TMEM_HELPERS_H
// =============================================================================
// tmem_helpers.h — shared TMEM test helpers for tcgen05 integration tests.
//
// Extracted from tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp
// per FU-5 Phase 1 refactoring (openspec/changes/tcgen05-flashattention-coverage).
//
// Functions:
//   - fill_tmem_with_golden_inputs(tmem)  — populate A/B slots with known f16 values
//   - require_c_slot_matches(tmem, expected, context) — verify C slots match golden
//   - compare_c_slot_to_reference(tmem, expected, eps, margin, ctx) — flexible check
//   - read_c_fragments(tmem, base_slot) — read C fragment floats as array
//
// A/B fragment layout (per tcgen05_helpers.h:20-21):
//   - lane_id reads A from slot lane_id*2
//   - lane_id reads B from slot lane_id*2+1
// C fragment layout (per tcgen05_helpers.h:22-24):
//   - warp owns C slots [warp_id*32+64 .. warp_id*32+95], 32 slots per warp
//   - warp 0: [64..95], warp 1: [96..127]
//
// DEPENDENCIES: Callers must #include "catch_amalgamated.hpp" BEFORE this
// header to make Catch2 macros (INFO, REQUIRE) available.
// =============================================================================

#include "ptxsim/memory/tmem.h"
#include "ptxsim/utils/half_utils.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

namespace ptxsim {
namespace testing {
namespace tmem {

// =========================================================================
// fill_tmem_with_golden_inputs
//
// Populate TMEM A fragments (slots 0,2,4,...,62) and B fragments (slots
// 1,3,5,...,63) with hand-computed golden f16 inputs.
//
// Golden inputs (per tcgen05_mma_golden.h):
//   A: 8×8 f16 matrix; A[i][0]=i+1, A[i][k>0]=0 → writes f16 at index i*8
//   B: 8×4 f16 matrix; B[0][j]=j+1, B[k>0][j]=0 → writes f16 at index j
//
// These inputs, when summed over k (MMA outer product), produce:
//   C[i][j] = A[i][0] * B[0][j] = (i+1)*(j+1)
//
// UNVERIFIED-AGAINST-HARDWARE — hand-computed per PTX ISA §9.7.16.
// =========================================================================
inline void fill_tmem_with_golden_inputs(Tmem &tmem) {
    // A slot pattern: 8 rows × 8 cols, only col 0 nonzero.
    std::array<uint8_t, Tmem::kSlotSize> a_buf{};
    for (int i = 0; i < 8; ++i) {
        const uint16_t h = f32_to_f16(static_cast<float>(i + 1));
        const size_t byte_idx = static_cast<size_t>(i) * 8 * 2;
        a_buf[byte_idx]     = static_cast<uint8_t>(h & 0xFF);
        a_buf[byte_idx + 1] = static_cast<uint8_t>(h >> 8);
    }

    // B slot pattern: 8 rows × 4 cols, only row 0 nonzero.
    std::array<uint8_t, Tmem::kSlotSize> b_buf{};
    for (int j = 0; j < 4; ++j) {
        const uint16_t h = f32_to_f16(static_cast<float>(j + 1));
        b_buf[j * 2]     = static_cast<uint8_t>(h & 0xFF);
        b_buf[j * 2 + 1] = static_cast<uint8_t>(h >> 8);
    }

    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        tmem.write(static_cast<size_t>(lane_id) * 2, a_buf.data(),
                   Tmem::kSlotSize);
        tmem.write(static_cast<size_t>(lane_id) * 2 + 1, b_buf.data(),
                   Tmem::kSlotSize);
    }
}

// =========================================================================
// read_c_fragments
//
// Read C fragments from slots [base_slot..base_slot+31] as 32 f32 values
// per lane (8 rows × 4 cols, row-major: index = i*4+j). Returns a
// 32×32 float matrix (lane_id × fragment_idx).
// =========================================================================
inline std::array<std::array<float, 32>, 32> read_c_fragments(
    Tmem &tmem, size_t base_slot) {
    std::array<std::array<float, 32>, 32> result{};
    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        size_t slot = base_slot + static_cast<size_t>(lane_id);
        std::array<uint8_t, Tmem::kSlotSize> buf{};
        tmem.read(slot, buf.data(), Tmem::kSlotSize);
        std::memcpy(result[lane_id].data(), buf.data(),
                    sizeof(float) * 32);
    }
    return result;
}

// =========================================================================
// require_c_slot_matches
//
// Strict equivalence check: verifies TMEM C slots [64..95] (warp 0) match
// the expected 32-element f32 array across all 32 lanes. Uses Catch2
// REQUIRE/INFO macros — caller MUST include "catch_amalgamated.hpp" first.
// epsilon=1e-6 for f32 equality.
// =========================================================================
inline void require_c_slot_matches(Tmem &tmem,
                                   const std::array<float, 32> &expected,
                                   const char *context_info) {
    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        std::array<uint8_t, Tmem::kSlotSize> c_buf{};
        tmem.read(static_cast<size_t>(64) + static_cast<size_t>(lane_id),
                  c_buf.data(), Tmem::kSlotSize);

        alignas(16) float c_arr[32];
        std::memcpy(c_arr, c_buf.data(), sizeof(c_arr));

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 4; ++j) {
                const int idx = i * 4 + j;
                const float actual = c_arr[idx];
                INFO(context_info << " lane=" << lane_id << " i=" << i
                     << " j=" << j << " expected=" << expected[idx]
                     << " actual=" << actual);
                REQUIRE(actual == Catch::Approx(expected[idx]).epsilon(1e-6f));
            }
        }
    }
}

// =========================================================================
// compare_c_slot_to_reference
//
// Flexible comparison: verifies TMEM C slots [64..95] match expected
// within user-specified tolerance. Returns true if all 32×32 elements
// pass. For K=128 accumulator tests where error accumulates.
//
// Parameters:
//   - expected: 32 f32 reference values
//   - epsilon: relative tolerance (e.g., 1e-3 for K=128 accumulation)
//   - margin: absolute tolerance margin (e.g., 1e-5 for near-zero)
//   - context_info: diagnostic label
// =========================================================================
inline bool compare_c_slot_to_reference(
    Tmem &tmem,
    const std::array<float, 32> &expected,
    double epsilon, double margin,
    const char *context_info) {
    bool all_passed = true;
    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        std::array<uint8_t, Tmem::kSlotSize> c_buf{};
        tmem.read(static_cast<size_t>(64) + static_cast<size_t>(lane_id),
                  c_buf.data(), Tmem::kSlotSize);

        alignas(16) float c_arr[32];
        std::memcpy(c_arr, c_buf.data(), sizeof(c_arr));

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 4; ++j) {
                const int idx = i * 4 + j;
                const float actual = c_arr[idx];
                const float ref = expected[idx];
                const float diff = std::abs(actual - ref);
                const float allowed = static_cast<float>(
                    epsilon * std::abs(ref) + margin);
                if (diff > allowed) {
                    INFO(context_info << " MISMATCH lane=" << lane_id
                         << " i=" << i << " j=" << j
                         << " expected=" << ref << " actual=" << actual
                         << " diff=" << diff << " allowed=" << allowed);
                    all_passed = false;
                }
            }
        }
    }
    return all_passed;
}

} // namespace tmem
} // namespace testing
} // namespace ptxsim

#endif // PTXSIM_TESTING_TMEM_HELPERS_H