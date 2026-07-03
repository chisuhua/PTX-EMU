// test_participation_mask_boundaries.cpp
//
// Direct unit tests for WarpBarrier participation_mask boundary conditions.
//
// Coverage gap closed:
//   - tests/unit/exec/test_exec_layer_e1_e3.cpp::E2a-E2d (Phase 7 Wbar deletion
//     removed 32-thread full mask + 16-thread partial mask boundary tests)
//   - tests/unit/barrier/test_barrier_verification.cpp (-162 lines, Phase 7)
//
// These tests directly assert WarpBarrier::is_complete() respects
// participation_mask exactly: a barrier is complete only when ALL participants
// in participation_mask have arrived, not when the full 32-lane set has arrived.
//
// Spec: openspec/changes/barrier-module-lifecycle-tests/specs/barrier-module-unit-tests/spec.md
//   "participation_mask boundary conditions MUST be respected"

#include "catch_amalgamated.hpp"
#include "ptxsim/barrier/warp_barrier.h"

using namespace ptxsim;

TEST_CASE("WarpBarrier participation_mask boundaries",
          "[barrier][warp_barrier][participation_mask][boundaries]") {

    SECTION("full_mask_32_arrive_31_is_incomplete: "
            "32-bit mask with 31 arrivals MUST NOT complete") {
        WarpBarrier wb;
        wb.init(0xFFFFFFFFu, 21, 20);

        REQUIRE(wb.get_expected_count() == 32);
        REQUIRE(wb.get_arrived_count() == 0);
        REQUIRE_FALSE(wb.is_complete());

        // Arrive 31 lanes (0..30)
        for (int i = 0; i < 31; ++i) {
            wb.arrive(i);
        }

        // Key assertion: 31 arrivals on 32-bit mask MUST NOT be complete
        CHECK(wb.get_arrived_count() == 31);
        CHECK(wb.get_arrived_mask() == 0x7FFFFFFFu);
        CHECK_FALSE(wb.is_complete());
        CHECK(wb.get_missing_mask() == 0x80000000u);  // lane 31 missing
    }

    SECTION("partial_mask_16_all_arrive_completes_at_16: "
            "16-bit mask MUST complete at 16 arrivals (not 32)") {
        WarpBarrier wb;
        wb.init(0x0000FFFFu, 21, 20);

        REQUIRE(wb.get_expected_count() == 16);
        REQUIRE(wb.get_arrived_count() == 0);
        REQUIRE_FALSE(wb.is_complete());

        // Arrive exactly the 16 participants
        for (int i = 0; i < 16; ++i) {
            wb.arrive(i);
        }

        // Key assertion: 16 arrivals on 16-bit mask MUST complete
        CHECK(wb.get_arrived_count() == 16);
        CHECK(wb.get_arrived_mask() == 0x0000FFFFu);
        CHECK(wb.is_complete());
        CHECK(wb.get_state() == WarpBarrier::State::Complete);
        CHECK(wb.get_missing_mask() == 0u);
    }
}
