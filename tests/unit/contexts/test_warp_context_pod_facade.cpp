// Unit tests for T2-3 A4a — WarpContext POD facade: header-level
// verifies that WarpContext holds 3 POD members (lane_mask_, warp_identity_,
// backend_links_) with default-initialized values.
//
// A4a is purely additive — old fields remain the canonical source.
// These tests ONLY verify the POD members exist with defaults.

#include "ptxsim/contexts/backend_links.h"
#include "ptxsim/contexts/lane_mask.h"
#include "ptxsim/contexts/warp_identity.h"
#include "ptxsim/warp_context.h"
#include <catch_amalgamated.hpp>

using ptxsim::contexts::BackendLinksPod;
using ptxsim::contexts::LaneMaskPod;
using ptxsim::contexts::WarpIdentityPod;

TEST_CASE("WarpContext facade: 3 POD members exist with default values",
          "[warp_context][facade][pod]") {
    WarpContext warp;

    // LaneMaskPod: defaults
    REQUIRE(warp.lane_mask_.active_mask[0] == false);
    REQUIRE(warp.lane_mask_.active_count == 0);
    REQUIRE(warp.lane_mask_.divergence_detected == false);
    REQUIRE(warp.lane_mask_.is_scheduled_ == false);
    REQUIRE(warp.lane_mask_.warp_thread_ids[0] == 0);

    // WarpIdentityPod: defaults
    REQUIRE(warp.warp_identity_.warp_id == 0);
    REQUIRE(warp.warp_identity_.physical_warp_id == 0);
    REQUIRE(warp.warp_identity_.physical_block_id == 0);
    REQUIRE(warp.warp_identity_.pc == 0);

    // BackendLinksPod: defaults
    REQUIRE(warp.backend_links_.register_bank_manager_ == nullptr);
    REQUIRE(warp.backend_links_.sm_context_ == nullptr);
    REQUIRE(warp.backend_links_.cta_context_ == nullptr);
    REQUIRE(warp.backend_links_.threads.empty());
    REQUIRE(warp.backend_links_.single_step_mode == false);
}