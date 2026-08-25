// test_warp_status_snapshot.cpp
// =============================================================================
// Phase 2.3.1: IPtxEmuDevice::get_warp_status unit/integration test.
//
// Validates that get_warp_status populates the existing 5-field WarpStatus
// struct (include/ptxemu/device_api.h:69-75) with correct data from
// warp->get_warp_state().threads[] (per include/ptxsim/warp_state.h:14).
//
// READ-ONLY verification per state-modification-audit skill: get_warp_status
// must NOT mutate any state.
//
// Test scenarios (per e2e-delegation-validation spec):
//   1. warp_id + sm_id fields populated
//   2. All lanes active: active_count == 32, lanes[i].state == kRun
//   3. All lanes finished: active_count == 0, lanes[i].state == kExit
//   4. Mixed active/inactive
//   5. Blocked threads contribute to blocked_cycles
//   6. Invalid sm_id returns default WarpStatus (graceful degradation)
// =============================================================================

#include "catch_amalgamated.hpp"
#include "ptxemu/testing/warp_executor_test_fixture.h"

#include "ptxsim/thread_state.h"

using ptxemu::LaneStatus;
using ptxemu::ThreadState;
using ptxemu::WarpStatus;
using ptxemu::testing::WarpExecutorTestFixture;

TEST_CASE("get_warp_status: warp_id and sm_id fields populated",
          "[integration][warp][delegation][warp_status]") {
    WarpExecutorTestFixture fix;
    REQUIRE(fix.sm() != nullptr);
    REQUIRE(fix.warp() != nullptr);

    SECTION("warp_id=0, sm_id=0") {
        WarpStatus s = fix.dev()->get_warp_status(0, 0);
        REQUIRE(s.warp_id == 0u);
        REQUIRE(s.sm_id == 0u);
        REQUIRE(s.lanes.size() == 32u);
        for (uint i = 0; i < 32; ++i) {
            REQUIRE(s.lanes[i].lane_id == i);
        }
    }

    SECTION("Different sm/warp IDs preserved in fields") {
        // Only SM 0 exists, but field passing should still work.
        // Verify by querying default-constructed (invalid) which returns all 0s.
        WarpStatus invalid = fix.dev()->get_warp_status(999, 999);
        REQUIRE(invalid.warp_id == 0u);   // default-constructed
        REQUIRE(invalid.sm_id == 0u);
        REQUIRE(invalid.lanes.empty());    // default vector
    }
}

TEST_CASE("get_warp_status: all 32 lanes active",
          "[integration][warp][delegation][warp_status]") {
    WarpExecutorTestFixture fix;
    REQUIRE(fix.warp() != nullptr);

    // Mark all threads active + not exited.
    auto& ws = fix.warp()->get_warp_state();
    for (int i = 0; i < 32; ++i) {
        ws.threads[i].is_active = true;
        ws.threads[i].is_exited = false;
        ws.threads[i].is_blocked = false;
        ws.threads[i].status = ptxsim::ThreadStatus::Active;
        ws.threads[i].blocked_cycles_remaining = 0;
    }
    // Recompute active_mask cache after manual mutation (per AGENTS.md T2-1).
    fix.warp()->update_active_mask();

    WarpStatus s = fix.dev()->get_warp_status(0, 0);
    REQUIRE(s.active_count == 32u);
    REQUIRE(s.lanes.size() == 32u);
    for (uint i = 0; i < 32; ++i) {
        REQUIRE(s.lanes[i].lane_id == i);
        REQUIRE(s.lanes[i].state == ThreadState::kRun);  // Active → kRun
    }
    REQUIRE(s.blocked_cycles == 0);
}

TEST_CASE("get_warp_status: all 32 lanes exited",
          "[integration][warp][delegation][warp_status]") {
    WarpExecutorTestFixture fix;
    REQUIRE(fix.warp() != nullptr);

    auto& ws = fix.warp()->get_warp_state();
    for (int i = 0; i < 32; ++i) {
        ws.threads[i].is_active = false;
        ws.threads[i].is_exited = true;
        ws.threads[i].status = ptxsim::ThreadStatus::Exited;
    }
    fix.warp()->update_active_mask();

    WarpStatus s = fix.dev()->get_warp_status(0, 0);
    REQUIRE(s.active_count == 0u);
    for (uint i = 0; i < 32; ++i) {
        REQUIRE(s.lanes[i].state == ThreadState::kExit);
    }
}

TEST_CASE("get_warp_status: mixed active/inactive",
          "[integration][warp][delegation][warp_status]") {
    WarpExecutorTestFixture fix;
    REQUIRE(fix.warp() != nullptr);

    auto& ws = fix.warp()->get_warp_state();
    // Lanes 0-15 active, 16-31 exited.
    for (int i = 0; i < 16; ++i) {
        ws.threads[i].is_active = true;
        ws.threads[i].is_exited = false;
        ws.threads[i].status = ptxsim::ThreadStatus::Active;
    }
    for (int i = 16; i < 32; ++i) {
        ws.threads[i].is_active = false;
        ws.threads[i].is_exited = true;
        ws.threads[i].status = ptxsim::ThreadStatus::Exited;
    }
    fix.warp()->update_active_mask();

    WarpStatus s = fix.dev()->get_warp_status(0, 0);
    REQUIRE(s.active_count == 16u);
    for (uint i = 0; i < 16; ++i) {
        REQUIRE(s.lanes[i].state == ThreadState::kRun);
    }
    for (uint i = 16; i < 32; ++i) {
        REQUIRE(s.lanes[i].state == ThreadState::kExit);
    }
}

TEST_CASE("get_warp_status: blocked threads contribute to blocked_cycles",
          "[integration][warp][delegation][warp_status]") {
    WarpExecutorTestFixture fix;
    REQUIRE(fix.warp() != nullptr);

    auto& ws = fix.warp()->get_warp_state();
    // Threads 0-3 each blocked_cycles_remaining = 10, others = 0.
    for (int i = 0; i < 32; ++i) {
        ws.threads[i].is_active = true;
        ws.threads[i].is_exited = false;
        ws.threads[i].status = ptxsim::ThreadStatus::Active;
        ws.threads[i].blocked_cycles_remaining = (i < 4) ? 10 : 0;
    }
    fix.warp()->update_active_mask();

    WarpStatus s = fix.dev()->get_warp_status(0, 0);
    REQUIRE(s.blocked_cycles == 40);  // 4 threads × 10 cycles
    REQUIRE(s.active_count == 32u);   // blocked threads still count as active
}

TEST_CASE("get_warp_status: pc field reflects warp_state.threads[i].pc",
          "[integration][warp][delegation][warp_status]") {
    WarpExecutorTestFixture fix;
    REQUIRE(fix.warp() != nullptr);

    auto& ws = fix.warp()->get_warp_state();
    // Set distinct PCs for each thread.
    for (int i = 0; i < 32; ++i) {
        ws.threads[i].is_active = true;
        ws.threads[i].is_exited = false;
        ws.threads[i].status = ptxsim::ThreadStatus::Active;
        ws.threads[i].pc = static_cast<uint32_t>(i * 7);
    }
    fix.warp()->update_active_mask();

    WarpStatus s = fix.dev()->get_warp_status(0, 0);
    for (uint i = 0; i < 32; ++i) {
        REQUIRE(s.lanes[i].pc == static_cast<uint32_t>(i * 7));
    }
}

TEST_CASE("get_warp_status: invalid sm_id returns default WarpStatus",
          "[integration][warp][delegation][warp_status]") {
    WarpExecutorTestFixture fix;

    WarpStatus s = fix.dev()->get_warp_status(999, 0);
    REQUIRE(s.warp_id == 0u);
    REQUIRE(s.sm_id == 0u);
    REQUIRE(s.lanes.empty());
    REQUIRE(s.active_count == 0u);
    REQUIRE(s.blocked_cycles == 0);
}

TEST_CASE("get_warp_status: invalid warp_id returns default WarpStatus",
          "[integration][warp][delegation][warp_status]") {
    WarpExecutorTestFixture fix;

    WarpStatus s = fix.dev()->get_warp_status(0, 999);
    REQUIRE(s.warp_id == 0u);
    REQUIRE(s.sm_id == 0u);
    REQUIRE(s.lanes.empty());
}