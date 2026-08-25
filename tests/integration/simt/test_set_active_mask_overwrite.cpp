// test_set_active_mask_overwrite.cpp
// =============================================================================
// Phase 2.2 regression guard: IPtxEmuDevice::set_active_mask OVERWRITE semantics.
//
// BUG-RETHANG / BUG-POSTBARRIER-TWOHALVES regression vector (per ptx-lessons-learned §1):
//   If set_active_mask is reimplemented with OR-merge instead of overwrite,
//   the ret handler's lane-clearing semantics breaks. After barrier release,
//   active_mask must reflect the latest set_active_mask call, NOT a
//   cumulative OR of historical masks.
//
// Test strategy (per OpenSpec phase-2-2-1-3-1-followup §3.6):
//   - Setup: g_gpu_context + 1 SM with 1 warp (via WarpExecutorTestFixture)
//   - Verify pre-condition: warp.active_mask_ == 0xFF (all lanes)
//   - Call: dev->set_active_mask(0, 0, 0x01)
//   - Verify: warp.active_mask_ == 0x01 (overwrite)
//   - NOT expected: 0xFF (no-op) or 0xFF | 0x01 = 0xFF (OR-merge)
//
// Phase 2.3.1 update: replaced WARN+early-return guard with proper warp
// setup via WarpExecutorTestFixture (shared in tests/integration/warp/).
//
// This integration test exercises the full delegation path through
// IPtxEmuDevice → WarpContext, validating that the IPtxEmuDevice layer
// preserves WarpContext overwrite semantics.
//
// Ref: openspec/changes/device-api-delegation/specs/ptxemu-device-api-delegation/spec.md
// Ref: openspec/changes/phase-2-2-1-3-1-followup/proposal.md §3.6
// Ref: ptx-barrier-mechanism skill (set_active_mask overwrite semantics)
// =============================================================================

#include "catch_amalgamated.hpp"
#include "ptxemu/testing/warp_executor_test_fixture.h"

#include <memory>

using ptxemu::testing::WarpExecutorTestFixture;

TEST_CASE("set_active_mask: overwrite (BUG-RETHANG regression guard)",
          "[integration][simt][delegation][regression]") {
    WarpExecutorTestFixture scope;
    REQUIRE(scope.gpu() != nullptr);
    REQUIRE(scope.sm() != nullptr);
    REQUIRE(scope.warp() != nullptr);
    REQUIRE(scope.dev() != nullptr);

    SECTION("starting mask 0xFF, set_active_mask(0x01) → mask becomes 0x01 (overwrite)") {
        // Pre-condition: all lanes active.
        scope.warp()->set_active_mask(0xFFFFFFFFu);
        REQUIRE(scope.warp()->get_active_mask() == 0xFFFFFFFFu);

        // Call delegation method.
        bool result = scope.dev()->set_active_mask(0, 0, 0x01u);
        REQUIRE(result == true);

        // Verify OVERWRITE (NOT OR-merge, NOT no-op).
        REQUIRE(scope.warp()->get_active_mask() == 0x01u);
    }

    SECTION("starting mask 0x00, set_active_mask(0xFF) → mask becomes 0xFF (overwrite)") {
        scope.warp()->set_active_mask(0x00000000u);
        REQUIRE(scope.warp()->get_active_mask() == 0x00000000u);

        bool result = scope.dev()->set_active_mask(0, 0, 0xFFu);
        REQUIRE(result == true);

        REQUIRE(scope.warp()->get_active_mask() == 0xFFu);
    }

    SECTION("set_active_mask twice → second wins (last-write-wins overwrite)") {
        scope.warp()->set_active_mask(0xFFFFFFFFu);
        scope.dev()->set_active_mask(0, 0, 0x0Fu);
        REQUIRE(scope.warp()->get_active_mask() == 0x0Fu);

        // Second call should overwrite the first, NOT OR-merge (0x0F | 0xF0 = 0xFF).
        scope.dev()->set_active_mask(0, 0, 0xF0u);
        REQUIRE(scope.warp()->get_active_mask() == 0xF0u);
    }
}

TEST_CASE("set_active_mask: invalid sm_id returns false without crash",
          "[integration][simt][delegation]") {
    WarpExecutorTestFixture scope;
    REQUIRE(scope.dev() != nullptr);

    // Invalid sm_id (e.g., 999 when GPU only has 1 SM) must return false
    // without crashing.
    bool result = scope.dev()->set_active_mask(999, 0, 0x01u);
    REQUIRE(result == false);
}

TEST_CASE("set_next_pc: invalid sm_id returns false without crash",
          "[integration][simt][delegation]") {
    WarpExecutorTestFixture scope;
    REQUIRE(scope.dev() != nullptr);

    bool result = scope.dev()->set_next_pc(999, 0, 0, 42u);
    REQUIRE(result == false);
}

TEST_CASE("set_active_mask + BarrierModule interaction: overwrite observable in barrier",
          "[integration][simt][delegation][regression][barrier]") {
    // Phase 2.3.1 addition: verify BUG-POSTBARRIER-TWOHALVES guard holds.
    // After overwrite to lane 0 only, barrier arrival/release must observe
    // only lane 0 as active (not the original 0xFF).
    WarpExecutorTestFixture scope;
    REQUIRE(scope.warp() != nullptr);

    scope.warp()->set_active_mask(0xFFFFFFFFu);
    REQUIRE(scope.warp()->get_active_mask() == 0xFFFFFFFFu);

    // Overwrite to lane 0 only.
    bool ok = scope.dev()->set_active_mask(0, 0, 0x01u);
    REQUIRE(ok == true);
    REQUIRE(scope.warp()->get_active_mask() == 0x01u);

    // Verify the count_active_lanes() (used by barrier arrival accounting)
    // reflects the overwrite.
    auto& ws = scope.warp()->get_warp_state();
    int active_lanes = ws.count_active_lanes();
    REQUIRE(active_lanes == 1);  // Only lane 0 is active after overwrite.
}