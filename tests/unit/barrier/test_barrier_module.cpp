#include "catch_amalgamated.hpp"
#include "ptxsim/barrier/warp_barrier.h"
#include "ptxsim/barrier/cta_barrier.h"
#include "ptxsim/barrier/barrier_module.h"

using namespace ptxsim;

TEST_CASE("WarpBarrier initialization", "[barrier][warp_barrier]") {
    WarpBarrier wb;

    SECTION("Initial state is Uninitialized") {
        REQUIRE(wb.get_state() == WarpBarrier::State::Uninitialized);
        REQUIRE(wb.is_initialized() == false);
        REQUIRE(wb.get_participation_mask() == 0);
        REQUIRE(wb.get_arrived_mask() == 0);
        REQUIRE(wb.get_expected_count() == 0);
        REQUIRE(wb.get_arrived_count() == 0);
    }

    SECTION("Init sets participation mask and state") {
        wb.init(0xFFFF0000, 21, 20);
        REQUIRE(wb.is_initialized() == true);
        REQUIRE(wb.get_state() == WarpBarrier::State::Initializing);
        REQUIRE(wb.get_participation_mask() == 0xFFFF0000);
        REQUIRE(wb.get_expected_count() == 16);
        REQUIRE(wb.get_reconvergence_pc() == 21);
        REQUIRE(wb.get_barrier_pc() == 20);
    }

    SECTION("Reset returns to Uninitialized") {
        wb.init(0xFFFF, 10, 5);
        wb.reset();
        REQUIRE(wb.get_state() == WarpBarrier::State::Uninitialized);
        REQUIRE(wb.is_initialized() == false);
    }
}

TEST_CASE("WarpBarrier arrive and complete", "[barrier][warp_barrier]") {
    WarpBarrier wb;

    SECTION("Single thread arrive does not complete") {
        wb.init(0xFFFF, 10, 5);
        wb.arrive(0);
        REQUIRE(wb.get_arrived_count() == 1);
        REQUIRE(wb.is_complete() == false);
        REQUIRE(wb.needs_to_wait(0) == false);
    }

    SECTION("All threads arrive completes barrier") {
        wb.init(0x000F, 10, 5);
        for (int i = 0; i < 3; i++) {
            wb.arrive(i);
            REQUIRE(wb.is_complete() == false);
        }
        wb.arrive(3);
        REQUIRE(wb.is_complete() == true);
        REQUIRE(wb.get_state() == WarpBarrier::State::Complete);
    }

    SECTION("Duplicate arrive is ignored") {
        wb.init(0x000F, 10, 5);
        wb.arrive(0);
        wb.arrive(0);
        REQUIRE(wb.get_arrived_count() == 1);
    }

    SECTION("needs_to_wait for non-participant returns false after complete") {
        wb.init(0x000F, 10, 5);
        wb.arrive(0);
        wb.arrive(1);
        wb.arrive(2);
        wb.arrive(3);
        REQUIRE(wb.is_complete() == true);
        REQUIRE(wb.needs_to_wait(4) == false);
    }

    SECTION("get_missing_mask shows waiting threads") {
        wb.init(0x000F, 10, 5);
        wb.arrive(0);
        wb.arrive(1);
        REQUIRE(wb.get_missing_mask() == 0x000C);
    }
}

TEST_CASE("CTABarrier basic operations", "[barrier][cta_barrier]") {
    CTABarrier cb;

    SECTION("Initial state is not initialized") {
        REQUIRE(cb.get_arrived_count() == 0);
        REQUIRE(cb.is_complete() == false);
    }

    SECTION("Init sets expected thread count") {
        cb.init(0, 32, 1);
        REQUIRE(cb.get_expected_threads() == 32);
        REQUIRE(cb.get_warp_count() == 1);
    }
}

TEST_CASE("BarrierModule warp barrier management", "[barrier][barrier_module]") {
    BarrierModule bm;

    SECTION("Initially no active barriers") {
        REQUIRE(bm.get_active_warp_barrier_count() == 0);
        REQUIRE(bm.get_active_cta_barrier_count() == 0);
    }

    SECTION("Init warp barrier returns valid pointer") {
        WarpBarrier* wb = bm.init_warp_barrier(0, 0xFFFF, 10, 5);
        REQUIRE(wb != nullptr);
        REQUIRE(bm.get_active_warp_barrier_count() == 1);
    }

    SECTION("Get warp barrier returns same instance") {
        bm.init_warp_barrier(0, 0xFFFF, 10, 5);
        WarpBarrier* wb1 = bm.get_warp_barrier(0);
        WarpBarrier* wb2 = bm.get_warp_barrier(0);
        REQUIRE(wb1 == wb2);
    }

    SECTION("Get invalid warp barrier returns nullptr") {
        REQUIRE(bm.get_warp_barrier(-1) == nullptr);
        REQUIRE(bm.get_warp_barrier(4) == nullptr);
    }

    SECTION("arrive_at_warp_barrier returns false until complete") {
        bm.init_warp_barrier(0, 0x000F, 10, 5);
        bool complete = bm.arrive_at_warp_barrier(0, 0);
        REQUIRE(complete == false);
        complete = bm.arrive_at_warp_barrier(0, 1);
        REQUIRE(complete == false);
        complete = bm.arrive_at_warp_barrier(0, 2);
        REQUIRE(complete == false);
        complete = bm.arrive_at_warp_barrier(0, 3);
        REQUIRE(complete == true);
    }

    SECTION("warp_barrier_needs_wait after arrive") {
        bm.init_warp_barrier(0, 0x000F, 10, 5);
        bm.arrive_at_warp_barrier(0, 0);
        REQUIRE(bm.warp_barrier_needs_wait(0, 0) == false);
        REQUIRE(bm.warp_barrier_needs_wait(0, 1) == true);
        bm.arrive_at_warp_barrier(0, 1);
        REQUIRE(bm.warp_barrier_needs_wait(0, 1) == false);
    }

    SECTION("reset_all clears all barriers") {
        bm.init_warp_barrier(0, 0xFFFF, 10, 5);
        bm.init_cta_barrier(0, 32, 1);
        bm.reset_all();
        REQUIRE(bm.get_active_warp_barrier_count() == 0);
    }
}

// ============================================================================
// Regression: BUG-RECONVERGENCE-SIMPLEGEMM
// When a divergent warp hits a barrier in two halves (force_reconvergence
// path), the second half calls init_warp_barrier with a fresh
// participation_mask. The init MUST update metadata (participation_mask,
// reconvergence_pc, expected_count) but PRESERVE arrived_mask and
// arrived_count — otherwise the second half loses the first half's
// accumulation and is_complete() returns the wrong answer.
//
// Pre-fix:  re-init resets arrived_mask to 0 (loses first half)
// Post-fix:  re-init preserves arrived_mask (accumulates across re-inits)
// ============================================================================
TEST_CASE("WarpBarrier::init preserves arrived_mask on re-init (BUG-RECONVERGENCE-SIMPLEGEMM)",
          "[barrier][warp_barrier][init][regression][BUG-RECONVERGENCE-SIMPLEGEMM]")
{
    WarpBarrier wbar;

    // First init: first half (lanes 0-15) participates
    wbar.init(0x0000FFFFu, 21, 20);
    for (int i = 0; i < 16; ++i) {
        wbar.arrive(i);
    }
    REQUIRE(wbar.get_arrived_mask() == 0x0000FFFFu);
    REQUIRE(wbar.get_arrived_count() == 16);
    REQUIRE(wbar.is_complete());

    // Re-init: second half (lanes 16-31) — force_reconvergence path
    wbar.init(0xFFFF0000u, 21, 20);

    // KEY ASSERTION: arrived_mask MUST be preserved (not reset to 0)
    // Metadata MUST be updated to reflect new participation
    CHECK(wbar.get_arrived_mask() == 0x0000FFFFu);
    CHECK(wbar.get_arrived_count() == 16);
    CHECK(wbar.get_participation_mask() == 0xFFFF0000u);
    CHECK(wbar.get_expected_count() == 16);
    CHECK(wbar.get_reconvergence_pc() == 21);
    CHECK(wbar.is_initialized());

    // Second half arrives; expected_count is now 16, so 16 more lanes
    // arriving brings arrived_count to 32 — barrier is complete
    for (int i = 16; i < 32; ++i) {
        wbar.arrive(i);
    }
    CHECK(wbar.get_arrived_mask() == 0xFFFFFFFFu);
    CHECK(wbar.get_arrived_count() == 32);
    CHECK(wbar.is_complete());
}