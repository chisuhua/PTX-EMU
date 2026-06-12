// tests/unit/warp/test_warp_blocked_decrement.cpp
// Type 1 (unit) test for BUG-001: active_count staleness after
// WarpContext::decrement_blocked_cycles() drains blocked cycles.
//
// Bug description:
//   When a lane is blocked (is_blocked=true, blocked_cycles_remaining=N),
//   the SM scheduler sees is_active()=false (active_count=0) and skips the
//   warp. Once the cycles drain, decrement_blocked_cycles() unblocks the
//   lane and sets is_active=true, but it does NOT update active_count.
//   The scheduler continues to skip the warp indefinitely → hang.
//
//   See: .omo/notepads/fix-ldglobal-active-count-hang/learnings.md
//
// Fix (in sm_context.cpp, NOT in this test):
//   Call WarpContext::update_active_mask() after the decrement loop.
//
// Red→Green contract:
//   - Before fix: this test FAILS (active_count is stale at 0).
//   - After fix:  this test PASSES (active_count is recomputed).
//
// This test directly instantiates WarpContext, drives
// decrement_blocked_cycles() 5 times, and verifies the contract.

#include "catch_amalgamated.hpp"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/thread_state.h"

#include <map>
#include <memory>
#include <vector>

using namespace ptxsim;

namespace {

// Helper: build a WarpContext with a single thread at lane 0.
// Mirrors the pattern in tests/unit/warp/test_warp_context.cpp::test_warp_thread_addition.
//
// WarpContext is non-copyable (holds unique_ptr<ThreadContext>),
// so we hand back a heap-allocated instance.
std::unique_ptr<WarpContext> make_warp_with_one_thread() {
    auto warp = std::make_unique<WarpContext>();

    Dim3 blockIdx  = {0, 0, 0};
    Dim3 threadIdx = {0, 0, 0};
    Dim3 gridDim   = {1, 1, 1};
    Dim3 blockDim  = {32, 1, 1};

    std::vector<StatementContext> statements;
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    auto thread = std::make_unique<ThreadContext>();
    thread->init(blockIdx, threadIdx, gridDim, blockDim, statements,
                 &name2Sym, label2pc, nullptr /*name2Share*/, nullptr /*cta_ctx*/);

    warp->add_thread(std::move(thread), 0);

    // Sanity: one thread added → active_count == 1 (WarpContext constructor
    // starts at 0 and add_thread increments exactly once).
    REQUIRE(warp->get_active_count() == 1);
    REQUIRE(warp->is_active());
    return warp;
}

}  // namespace

TEST_CASE("BUG-001: decrement_blocked_cycles must leave warp schedulable",
          "[unit][warp][bug][hang_regression]") {
    auto warp_ptr = make_warp_with_one_thread();
    WarpContext& warp = *warp_ptr;

    // -------------------------------------------------------------
    // Phase 1: Block lane 0 for exactly 5 cycles (the LdHandler
    //          hang scenario from the bug report).
    // -------------------------------------------------------------
    WarpState& ws = warp.get_warp_state();
    ws.threads[0].is_blocked               = true;
    ws.threads[0].blocked_cycles_remaining = 5;

    // The scheduler reads active_count → call update_active_mask to
    // propagate the "blocked" state into active_count (mirrors the
    // real path in execute_warp_instruction).
    warp.update_active_mask();
    REQUIRE(warp.get_active_count() == 0);
    REQUIRE_FALSE(warp.is_active());

    // -------------------------------------------------------------
    // Phase 2: Drive 5 ticks of decrement_blocked_cycles()
    //          (the loop body extracted from sm_context.cpp:182-197).
    // -------------------------------------------------------------
    for (int i = 0; i < 5; ++i) {
        WarpContext::decrement_blocked_cycles(warp.get_warp_state());
    }

    // Sanity: the per-lane state is now restored — lane 0 is unblocked
    // and marked active. This is the part decrement_blocked_cycles
    // already gets right.
    REQUIRE_FALSE(ws.threads[0].is_blocked);
    REQUIRE(ws.threads[0].blocked_cycles_remaining == 0);
    REQUIRE(ws.threads[0].is_active);

    // -------------------------------------------------------------
    // Phase 3 (BUG EXPOSURE): active_count is STALE.
    //
    //   decrement_blocked_cycles unblocked the lane and set its
    //   is_active field, but never told WarpContext about it, so
    //   active_count is still 0. The scheduler will keep skipping
    //   this warp → hang.
    // -------------------------------------------------------------
    SECTION("bug manifestation: active_count stale after decrement") {
        // BUG: these are the assertions that should expose the failure
        // pre-fix. With the bug present, active_count stays at 0 and
        // is_active() returns false even though the lane is schedulable.
        CHECK(warp.get_active_count() == 1);   // FAILS pre-fix
        CHECK(warp.is_active());               // FAILS pre-fix
    }

    // -------------------------------------------------------------
    // Phase 4 (FIX PATH): update_active_mask() heals the staleness.
    // -------------------------------------------------------------
    SECTION("fix path: update_active_mask() restores active_count") {
        warp.update_active_mask();
        CHECK(warp.get_active_count() == 1);
        CHECK(warp.is_active());
        CHECK(warp.is_lane_active(0));
    }
}

// Additional defensive test: multiple lanes, mixed blocked counts.
TEST_CASE("BUG-001 multi-lane: partial block / partial unblock",
          "[unit][warp][bug][hang_regression]") {
    auto warp_ptr = make_warp_with_one_thread();
    WarpContext& warp = *warp_ptr;

    WarpState& ws = warp.get_warp_state();
    // Mark all 32 lanes blocked; lane 0 drains in 2 cycles, the rest in 5.
    for (int i = 0; i < 32; ++i) {
        ws.threads[i].is_blocked               = true;
        ws.threads[i].blocked_cycles_remaining = 5;
    }
    ws.threads[0].blocked_cycles_remaining = 2;

    warp.update_active_mask();
    REQUIRE(warp.get_active_count() == 0);
    REQUIRE_FALSE(warp.is_active());

    // Drive 2 cycles: lane 0 unblocks, the other 31 lanes stay blocked.
    WarpContext::decrement_blocked_cycles(ws);
    WarpContext::decrement_blocked_cycles(ws);

    // Per-lane state: lane 0 unblocked+active; lanes 1..31 still blocked.
    REQUIRE_FALSE(ws.threads[0].is_blocked);
    REQUIRE(ws.threads[0].is_active);
    REQUIRE(ws.threads[0].blocked_cycles_remaining == 0);
    REQUIRE(ws.threads[1].is_blocked);
    REQUIRE(ws.threads[1].blocked_cycles_remaining == 3);

    // BUG: active_count is still 0 (stale) — the scheduler sees this as
    // a dead warp even though one lane is already schedulable.
    // Pre-fix: FAILS. Post-fix: PASSES.
    CHECK(warp.get_active_count() == 1);
    CHECK(warp.is_active());
}
