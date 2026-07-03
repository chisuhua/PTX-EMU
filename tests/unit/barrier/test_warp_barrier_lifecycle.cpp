// test_warp_barrier_lifecycle.cpp
//
// Direct unit tests for WarpBarrier lifecycle semantics.
//
// Coverage gap closed:
//   - tests/unit/sync/test_syncthreads_test3_repro.cpp (Phase 7 Wbar deletion
//     removed "Wbar completes with 16-thread participation mask" lifecycle
//     + 190 lines of direct Wbar tests)
//   - tests/unit/barrier/test_barrier_verification.cpp (-162 lines, Phase 7)
//
// These tests directly assert:
//   1. init → arrive → is_complete → reset → re-init → arrive → is_complete
//      full lifecycle works
//   2. Multiple cycles of complete → reset → re-init don't leak state
//
// NOTE: BUG-RECONVERGENCE-SIMPLEGEMM re-init invariant (preserving arrived_mask
// on re-init) is already covered by
// tests/unit/barrier/test_barrier_module.cpp::WarpBarrier::init preserves
// arrived_mask on re-init. This file does NOT duplicate that test.
//
// Spec: openspec/changes/barrier-module-lifecycle-tests/specs/barrier-module-unit-tests/spec.md
//   "WarpBarrier::init MUST support lifecycle"

#include "catch_amalgamated.hpp"
#include "ptxsim/barrier/warp_barrier.h"

using namespace ptxsim;

TEST_CASE("WarpBarrier full lifecycle: init -> arrive -> complete -> reset -> re-init -> complete",
          "[barrier][warp_barrier][lifecycle][reset][reinit]") {

    WarpBarrier wb;
    const uint32_t mask = 0xFFFFFFFFu;
    const int reconv_pc = 21;
    const uint32_t barrier_pc = 20;

    SECTION("first cycle: init + arrive(0..31) completes") {
        wb.init(mask, reconv_pc, barrier_pc);
        REQUIRE(wb.is_initialized());
        REQUIRE(wb.get_state() == WarpBarrier::State::Initializing);
        REQUIRE(wb.get_expected_count() == 32);

        for (int i = 0; i < 32; ++i) {
            wb.arrive(i);
        }

        REQUIRE(wb.get_arrived_count() == 32);
        REQUIRE(wb.get_arrived_mask() == 0xFFFFFFFFu);
        REQUIRE(wb.is_complete());
        REQUIRE(wb.get_state() == WarpBarrier::State::Complete);
    }

    SECTION("reset() returns to Uninitialized with all state zeroed") {
        wb.init(mask, reconv_pc, barrier_pc);
        for (int i = 0; i < 32; ++i) {
            wb.arrive(i);
        }
        REQUIRE(wb.is_complete());

        wb.reset();

        // After reset, MUST be Uninitialized with clean state
        CHECK_FALSE(wb.is_initialized());
        CHECK(wb.get_state() == WarpBarrier::State::Uninitialized);
        CHECK(wb.get_arrived_count() == 0);
        CHECK(wb.get_arrived_mask() == 0u);
        CHECK(wb.get_participation_mask() == 0u);
        CHECK(wb.get_expected_count() == 0);
        CHECK(wb.get_reconvergence_pc() == -1);  // implementation: reset to -1 (sentinel)
        CHECK(wb.get_barrier_pc() == 0);         // implementation: reset to 0
    }

    SECTION("re-init after reset: fresh cycle produces complete again") {
        // First cycle
        wb.init(mask, reconv_pc, barrier_pc);
        for (int i = 0; i < 32; ++i) {
            wb.arrive(i);
        }
        REQUIRE(wb.is_complete());

        // Reset
        wb.reset();
        REQUIRE_FALSE(wb.is_initialized());

        // Re-init: fresh cycle
        wb.init(mask, reconv_pc, barrier_pc);
        REQUIRE(wb.is_initialized());
        CHECK(wb.get_arrived_count() == 0);
        CHECK(wb.get_arrived_mask() == 0u);
        CHECK_FALSE(wb.is_complete());

        for (int i = 0; i < 32; ++i) {
            wb.arrive(i);
        }

        CHECK(wb.get_arrived_count() == 32);
        CHECK(wb.get_arrived_mask() == 0xFFFFFFFFu);
        CHECK(wb.is_complete());
    }
}

TEST_CASE("WarpBarrier multiple_completion_cycles_no_state_leak",
          "[barrier][warp_barrier][lifecycle][state_leak][regression]") {

    WarpBarrier wb;
    const uint32_t mask = 0x0000FFFFu;  // 16-lane mask for speed
    const int reconv_pc = 21;
    const uint32_t barrier_pc = 20;

    SECTION("3 consecutive cycles: each starts with arrived_count=0") {
        for (int cycle = 0; cycle < 3; ++cycle) {
            // Init
            wb.init(mask, reconv_pc, barrier_pc);
            REQUIRE(wb.is_initialized());
            REQUIRE(wb.get_arrived_count() == 0);
            REQUIRE_FALSE(wb.is_complete());

            // Arrive all 16 participants
            for (int i = 0; i < 16; ++i) {
                wb.arrive(i);
            }
            REQUIRE(wb.get_arrived_count() == 16);
            REQUIRE(wb.is_complete());

            // Reset for next cycle
            wb.reset();
            REQUIRE_FALSE(wb.is_initialized());
            REQUIRE(wb.get_arrived_count() == 0);
            REQUIRE(wb.get_arrived_mask() == 0u);
        }
    }
}
