// test_barrier_module_release.cpp
//
// Direct unit tests for BarrierModule::release_warp_barrier state translation.
//
// Coverage gap closed:
//   - 5 state translation invariants in barrier_module.cpp:85-138:
//     (1) is_blocked = false for released threads
//     (2) status = ptxsim::ThreadStatus::Active
//     (3) is_active = true (required: get_lanes_by_pc() filters on is_active)
//     (4) OR-merge active_mask with arrived_mask (not overwrite)
//     (5) advance_thread_pc(i, reconv_pc) for released threads
//
// These invariants were the cross-module state translation locked in by
// BUG-RECONVERGENCE-SIMPLEGEMM / BUG-POSTBARRIER-TWOHALVES fixes
// (lessons-learned.md §1, §19). Without direct unit coverage, future
// refactors of release_warp_barrier might silently drop one of these.
//
// Approach: directly construct WarpContext (default ctor initializes
// warp_state.threads[32] + active_mask[32] + exec_mask=0xFFFFFFFF) and call
// BarrierModule::release_warp_barrier() without going through step_warp /
// SMContext / CTAContext.
//
// Spec: openspec/changes/barrier-module-lifecycle-tests/specs/barrier-module-unit-tests/spec.md
//   "BarrierModule::release_warp_barrier MUST update thread state fields atomically"

#include "catch_amalgamated.hpp"
#include "ptxsim/barrier/barrier_module.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"

using namespace ptxsim;

TEST_CASE("BarrierModule::release_warp_barrier OR-merges active_mask "
          "(not overwrite)",
          "[barrier][barrier_module][release][active_mask][OR][regression][BUG-POSTBARRIER-TWOHALVES]") {
    // Setup: WarpContext with active_mask=0xFFFF0000 (upper 16 lanes already
    // active from a prior release). Lower 16 lanes are inactive.
    WarpContext warp_ctx;
    warp_ctx.set_active_mask(0xFFFF0000u);
    REQUIRE(warp_ctx.get_active_mask() == 0xFFFF0000u);

    // Now a new barrier releases only the lower 16 lanes (arrived_mask=0x0000FFFF)
    BarrierModule bm;
    bm.init_warp_barrier(0, 0x0000FFFFu, 21, 20);
    for (int i = 0; i < 16; ++i) {
        bm.arrive_at_warp_barrier(0, i);
    }
    REQUIRE(bm.is_warp_barrier_complete(0));

    // Release — should OR-merge, NOT overwrite
    bm.release_warp_barrier(0, &warp_ctx);

    // Both halves MUST remain active
    CHECK(warp_ctx.get_active_mask() == 0xFFFFFFFFu);
}

TEST_CASE("BarrierModule::release_warp_barrier resets is_blocked + status + is_active",
          "[barrier][barrier_module][release][state_translation][regression][lessons_§1]") {
    // Setup: WarpContext with all 16 participants marked as blocked
    WarpContext warp_ctx;
    warp_ctx.set_active_mask(0x0000FFFFu);  // lower 16 active

    // Pre-state: all 16 released threads are in blocked state
    for (int i = 0; i < 16; ++i) {
        auto& ts = warp_ctx.get_warp_state().threads[i];
        ts.is_blocked = true;
        ts.status = ptxsim::ThreadStatus::Blocked;
        ts.is_active = false;
        ts.pc = 20;  // barrier_pc
        ts.next_pc = 20;
    }
    // Sanity: pre-state is correct
    for (int i = 0; i < 16; ++i) {
        auto& ts = warp_ctx.get_warp_state().threads[i];
        REQUIRE(ts.is_blocked);
        REQUIRE(ts.status == ptxsim::ThreadStatus::Blocked);
        REQUIRE_FALSE(ts.is_active);
    }

    // Arrive all 16 lanes and release
    BarrierModule bm;
    bm.init_warp_barrier(0, 0x0000FFFFu, 21, 20);
    for (int i = 0; i < 16; ++i) {
        bm.arrive_at_warp_barrier(0, i);
    }
    REQUIRE(bm.is_warp_barrier_complete(0));

    bm.release_warp_barrier(0, &warp_ctx);

    // Post-state: all 5 translations applied
    for (int i = 0; i < 16; ++i) {
        auto& ts = warp_ctx.get_warp_state().threads[i];
        INFO("lane " << i);

        // (1) is_blocked = false
        CHECK_FALSE(ts.is_blocked);
        // (2) status = Active
        CHECK(ts.status == ptxsim::ThreadStatus::Active);
        // (3) is_active = true (required: get_lanes_by_pc() filters on this)
        CHECK(ts.is_active);
        // (5) advance_thread_pc to reconv_pc (21)
        CHECK(ts.pc == 21);
        CHECK(ts.next_pc == 21);
    }
}

TEST_CASE("BarrierModule::release_warp_barrier two-cycle OR preserves first half lanes "
          "(BUG-POSTBARRIER-TWOHALVES)",
          "[barrier][barrier_module][release][two_halves][regression][BUG-POSTBARRIER-TWOHALVES]") {
    // This is the canonical BUG-POSTBARRIER-TWOHALVES scenario:
    //   - A divergent warp's two halves hit the same bar.warp.sync at different
    //     times. Each half triggers its own init/arrive/complete/release cycle
    //     (NOT a single cycle with two releases — release_warp_barrier calls
    //     wbar.reset() at the end so the same barrier cannot be released twice).
    //   - The second release MUST OR-merge the new arrived_mask with the
    //     existing active_mask, otherwise the first half's lanes are LOST.
    //
    // Without the fix: active_mask becomes 0xFFFF0000 (only second half) after
    // the second release, losing the first half (0x0000FFFF) → scheduler
    // permanently skips the first half → divergence is broken.

    WarpContext warp_ctx;
    warp_ctx.set_active_mask(0u);  // initially no lanes active

    BarrierModule bm;

    // --- Cycle 1: first half (lanes 0-15) hit the barrier first ---
    bm.init_warp_barrier(0, 0x0000FFFFu, 21, 20);
    REQUIRE(bm.get_warp_barrier(0)->get_state() == WarpBarrier::State::Initializing);

    for (int i = 0; i < 16; ++i) {
        bm.arrive_at_warp_barrier(0, i);
    }
    REQUIRE(bm.is_warp_barrier_complete(0));

    bm.release_warp_barrier(0, &warp_ctx);

    // After cycle 1: only first half in active_mask
    CHECK(warp_ctx.get_active_mask() == 0x0000FFFFu);

    // --- Cycle 2: second half (lanes 16-31) hit the barrier later ---
    // (barrier is reset after cycle 1; init fresh)
    bm.init_warp_barrier(0, 0xFFFF0000u, 21, 20);
    REQUIRE(bm.get_warp_barrier(0)->get_state() == WarpBarrier::State::Initializing);
    REQUIRE(bm.get_warp_barrier(0)->get_arrived_count() == 0);  // fresh

    for (int i = 16; i < 32; ++i) {
        bm.arrive_at_warp_barrier(0, i);
    }
    REQUIRE(bm.is_warp_barrier_complete(0));

    bm.release_warp_barrier(0, &warp_ctx);

    // KEY ASSERTION (the bug fix): active_mask MUST be 0xFFFFFFFF (OR-merge)
    // Pre-fix: would be 0xFFFF0000 (overwrite, first half lost)
    CHECK(warp_ctx.get_active_mask() == 0xFFFFFFFFu);
}
