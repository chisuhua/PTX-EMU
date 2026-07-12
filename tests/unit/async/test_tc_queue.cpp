// tests/unit/async/test_tc_queue.cpp
// Phase 0.4 (Fix #8): Blackwell per-CTA async tensor core command queue.
//
// ≥10 TEST_CASEs cover default construction, commit-group monotonic counter,
// wait() lane blocking + completion_pc capture, commit() per-lane unblock
// (mirroring release_warp_barrier post-unblock pattern), multi-waiter
// ordering, clear reset, concurrent commit thread-safety, and
// completion_pc drift prevention.
// Consumed by tcgen05.* async commit/wait in Phase 1-3.

#include "catch_amalgamated.hpp"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>

#include "ptxsim/async/tc_queue.h"
#include "ptxsim/thread_state.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"

using group_id_t = TcQueue::group_id_t;

// ---------------------------------------------------------------------------
// 1. construct_default_counter_zero
// ---------------------------------------------------------------------------
TEST_CASE("construct_default_counter_zero", "[tc_queue][construct]") {
    TcQueue q;
    REQUIRE(q.current_counter() == 0);
    REQUIRE(q.pending_count() == 0);
}

// ---------------------------------------------------------------------------
// 2. commit_advances_counter_monotonically
// ---------------------------------------------------------------------------
TEST_CASE("commit_advances_counter_monotonically", "[tc_queue][commit]") {
    TcQueue q;
    q.commit(3);
    REQUIRE(q.current_counter() == 3);
    q.commit(2);
    REQUIRE(q.current_counter() == 3);  // no regression
    q.commit(7);
    REQUIRE(q.current_counter() == 7);
}

// ---------------------------------------------------------------------------
// 3. commit_cas_idempotent_same_group
// ---------------------------------------------------------------------------
TEST_CASE("commit_cas_idempotent_same_group", "[tc_queue][commit]") {
    TcQueue q;
    q.commit(5);
    REQUIRE(q.current_counter() == 5);
    q.commit(5);
    REQUIRE(q.current_counter() == 5);
}

// ---------------------------------------------------------------------------
// 4. wait_block_lane_at_pc
// ---------------------------------------------------------------------------
TEST_CASE("wait_block_lane_at_pc", "[tc_queue][wait]") {
    TcQueue q;
    WarpContext w;
    int lane = 3;
    uint32_t start_pc = 42;
    w.advance_thread_pc(lane, start_pc);

    q.wait(&w, lane, 5u);

    auto& ts = w.get_warp_state().threads[lane];
    REQUIRE(ts.is_blocked == true);
    REQUIRE(ts.status == ptxsim::ThreadStatus::Blocked);
}

// ---------------------------------------------------------------------------
// 5. wait_stores_completion_pc_plus_one_in_pending
// ---------------------------------------------------------------------------
TEST_CASE("wait_stores_completion_pc_plus_one_in_pending", "[tc_queue][wait]") {
    TcQueue q;
    WarpContext w;
    int lane = 7;
    w.advance_thread_pc(lane, 100);

    // Block the lane
    q.wait(&w, lane, 10u);
    REQUIRE(q.pending_count() == 1);

    // Release it — verify it resumes at pc+1 = 101
    q.commit(10);
    REQUIRE(w.get_thread_pc(lane) == 101);
}

// ---------------------------------------------------------------------------
// 6. wait_then_immediate_commit_releases_lane
// ---------------------------------------------------------------------------
TEST_CASE("wait_then_immediate_commit_releases_lane", "[tc_queue][wait][commit]") {
    TcQueue q;
    WarpContext w;
    int lane = 0;
    w.advance_thread_pc(lane, 200);

    // Lane waits for group 3
    q.wait(&w, lane, 3u);
    auto& ts = w.get_warp_state().threads[lane];
    REQUIRE(ts.is_blocked == true);

    // commit(1) should NOT release (1 < 3)
    q.commit(1);
    REQUIRE(ts.is_blocked == true);

    // commit(3) should release
    q.commit(3);
    REQUIRE(ts.is_blocked == false);
    REQUIRE(ts.status == ptxsim::ThreadStatus::Active);
    REQUIRE(ts.is_active == true);
    REQUIRE(w.get_thread_pc(lane) == 201);  // 200 + 1
}

// ---------------------------------------------------------------------------
// 7. wait_continues_blocked_until_commit_reaches_group
// ---------------------------------------------------------------------------
TEST_CASE("wait_continues_blocked_until_commit_reaches_group", "[tc_queue][wait][commit]") {
    TcQueue q;
    WarpContext w;
    int lane = 1;
    w.advance_thread_pc(lane, 50);

    q.wait(&w, lane, 5u);

    // commit(3) — still blocked
    q.commit(3);
    auto& ts = w.get_warp_state().threads[lane];
    REQUIRE(ts.is_blocked == true);
    REQUIRE(w.get_thread_pc(lane) == 50);  // PC unchanged

    // commit(5) — released
    q.commit(5);
    REQUIRE(ts.is_blocked == false);
    REQUIRE(ts.status == ptxsim::ThreadStatus::Active);
    REQUIRE(w.get_thread_pc(lane) == 51);  // 50 + 1
}

// ---------------------------------------------------------------------------
// 8. multiple_waiters_different_groups_release_in_order
// ---------------------------------------------------------------------------
TEST_CASE("multiple_waiters_different_groups_release_in_order", "[tc_queue][multi]") {
    TcQueue q;
    WarpContext w;
    int lane_a = 5, lane_b = 10;

    w.advance_thread_pc(lane_a, 300);
    w.advance_thread_pc(lane_b, 400);

    // lane_b waits for group 3, lane_a for group 5
    q.wait(&w, lane_a, 5u);  // lane 5, group 5
    q.wait(&w, lane_b, 3u);  // lane 10, group 3

    REQUIRE(q.pending_count() == 2);

    // commit(3) — only lane_b released
    q.commit(3);
    REQUIRE(w.get_warp_state().threads[lane_a].is_blocked == true);
    REQUIRE(w.get_warp_state().threads[lane_b].is_blocked == false);
    REQUIRE(w.get_thread_pc(lane_b) == 401);

    REQUIRE(q.pending_count() == 1);

    // commit(5) — lane_a released
    q.commit(5);
    REQUIRE(w.get_warp_state().threads[lane_a].is_blocked == false);
    REQUIRE(w.get_thread_pc(lane_a) == 301);
}

// ---------------------------------------------------------------------------
// 9. multiple_waiters_same_group_all_released
// ---------------------------------------------------------------------------
TEST_CASE("multiple_waiters_same_group_all_released", "[tc_queue][multi]") {
    TcQueue q;
    WarpContext w;
    const int kNumLanes = 3;
    int lanes[kNumLanes] = {2, 7, 15};

    for (int i = 0; i < kNumLanes; ++i) {
        w.advance_thread_pc(lanes[i], 600 + i);
        q.wait(&w, lanes[i], 5u);
    }
    REQUIRE(q.pending_count() == kNumLanes);

    q.commit(5);

    for (int i = 0; i < kNumLanes; ++i) {
        auto& ts = w.get_warp_state().threads[lanes[i]];
        REQUIRE(ts.is_blocked == false);
        REQUIRE(ts.status == ptxsim::ThreadStatus::Active);
        REQUIRE(ts.is_active == true);
        REQUIRE(w.get_thread_pc(lanes[i]) == 601 + i);
    }
    REQUIRE(q.pending_count() == 0);
}

// ---------------------------------------------------------------------------
// 10. clear_resets_to_initial_state
// ---------------------------------------------------------------------------
TEST_CASE("clear_resets_to_initial_state", "[tc_queue][clear]") {
    TcQueue q;
    WarpContext w;
    int lane = 1;
    w.advance_thread_pc(lane, 10);

    q.wait(&w, lane, 3u);
    q.commit(3);
    REQUIRE(q.pending_count() == 0);

    q.clear();
    REQUIRE(q.current_counter() == 0);
    REQUIRE(q.pending_count() == 0);
}

// ---------------------------------------------------------------------------
// 11. concurrent_commit_thread_safety
// ---------------------------------------------------------------------------
TEST_CASE("concurrent_commit_thread_safety", "[tc_queue][thread_safety]") {
    TcQueue q;
    const int kNumThreads = 4;
    const int kIterations = 100;

    auto worker = [&q](int tid) {
        for (int i = 0; i < kIterations; ++i) {
            q.commit(static_cast<uint64_t>(tid * 1000 + i));
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) {
        t.join();
    }

    // After all commits, counter should equal max value committed
    int max_id = (kNumThreads - 1) * 1000 + (kIterations - 1);
    REQUIRE(q.current_counter() >= static_cast<uint64_t>(max_id));
}

// ---------------------------------------------------------------------------
// 12. completion_pc_uses_stored_value_not_current
// ---------------------------------------------------------------------------
TEST_CASE("completion_pc_uses_stored_value_not_current", "[tc_queue][pc_drift]") {
    TcQueue q;
    WarpContext w;
    int lane = 3;
    w.advance_thread_pc(lane, 10);

    // Block at PC=10; completion_pc should be captured as 11
    q.wait(&w, lane, 1u);

    // Manually mutate pc AFTER blocking (simulates PC drift)
    w.advance_thread_pc(lane, 99);

    // Commit: should release to 11, NOT 100
    q.commit(1);
    REQUIRE(w.get_thread_pc(lane) == 11);
    REQUIRE(w.get_warp_state().threads[lane].is_blocked == false);
}

// ---------------------------------------------------------------------------
// 13. commit_releases_multiple_waiters_above_group_id
// ---------------------------------------------------------------------------
TEST_CASE("commit_releases_multiple_waiters_above_group_id", "[tc_queue][commit]") {
    TcQueue q;
    WarpContext w;

    w.advance_thread_pc(0, 10);
    w.advance_thread_pc(1, 20);

    q.wait(&w, 0, 3u);
    q.wait(&w, 1, 7u);
    REQUIRE(q.pending_count() == 2);

    // commit(10) — both should be released (3 ≤ 10, 7 ≤ 10)
    q.commit(10);
    REQUIRE(q.pending_count() == 0);
    REQUIRE(w.get_warp_state().threads[0].is_blocked == false);
    REQUIRE(w.get_warp_state().threads[1].is_blocked == false);
    REQUIRE(w.get_thread_pc(0) == 11);
    REQUIRE(w.get_thread_pc(1) == 21);
}

// ---------------------------------------------------------------------------
// 14. commit_mid_range_wakes_subset
// ---------------------------------------------------------------------------
TEST_CASE("commit_mid_range_wakes_subset", "[tc_queue][commit]") {
    TcQueue q;
    WarpContext w;

    w.advance_thread_pc(0, 1);
    w.advance_thread_pc(1, 2);
    w.advance_thread_pc(2, 3);

    q.wait(&w, 0, 1u);
    q.wait(&w, 1, 5u);
    q.wait(&w, 2, 10u);

    // commit(5) — lane 0 (group 1) and lane 1 (group 5) released
    q.commit(5);
    REQUIRE(q.pending_count() == 1);
    REQUIRE(w.get_warp_state().threads[0].is_blocked == false);
    REQUIRE(w.get_warp_state().threads[1].is_blocked == false);
    REQUIRE(w.get_warp_state().threads[2].is_blocked == true);

    // commit(10) — remaining released
    q.commit(10);
    REQUIRE(q.pending_count() == 0);
    REQUIRE(w.get_warp_state().threads[2].is_blocked == false);
}

// ---------------------------------------------------------------------------
// 15. wait_returns_immediately_when_counter_already_meets_group (§29)
// ---------------------------------------------------------------------------
TEST_CASE("wait_returns_immediately_when_counter_already_meets_group",
          "[tc_queue][wait][early_return][FU-1]") {
    TcQueue q;
    q.commit(2);

    WarpContext w;
    int lane = 5;
    w.advance_thread_pc(lane, 100);

    q.wait(&w, lane, 1u);

    REQUIRE(q.pending_count() == 0);
    auto& ts = w.get_warp_state().threads[lane];
    REQUIRE(ts.is_blocked == false);
}

// ---------------------------------------------------------------------------
// 16. wait_blocks_when_counter_below_group_then_commit_releases
// ---------------------------------------------------------------------------
TEST_CASE("wait_blocks_when_counter_below_group_then_commit_releases",
          "[tc_queue][wait][early_return][FU-1]") {
    TcQueue q;
    q.commit(1);

    WarpContext w;
    int lane = 3;
    w.advance_thread_pc(lane, 50);

    q.wait(&w, lane, 2u);
    REQUIRE(q.pending_count() == 1);
    REQUIRE(w.get_warp_state().threads[lane].is_blocked == true);

    q.commit(2);
    REQUIRE(q.pending_count() == 0);
    REQUIRE(w.get_warp_state().threads[lane].is_blocked == false);
    REQUIRE(w.get_thread_pc(lane) == 51);
}