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

TEST_CASE("WarpContext facade: add_thread() mirrors LaneMaskPod fields",
          "[warp_context][facade][pod][add_thread]") {
    WarpContext warp;

    // Set physical IDs (these also populate WarpIdentityPod, verified next)
    warp.set_physical_warp_id(7);
    warp.set_physical_block_id(2);

    // Build a minimal ThreadContext
    auto thread = std::make_unique<ThreadContext>();
    Dim3 blockIdx = {0, 0, 0};
    Dim3 threadIdx = {3, 0, 0}; // lane 3 in warp 0
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<StatementContext> stmts;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    thread->init(blockIdx, threadIdx, gridDim, blockDim, stmts, &name2Sym,
                 label2pc, nullptr, nullptr);

    warp.add_thread(std::move(thread), 3);

    // LaneMaskPod mirrored from legacy fields
    REQUIRE(warp.lane_mask_.warp_thread_ids[3] == 3);
    REQUIRE(warp.lane_mask_.active_mask[3] == true);
    REQUIRE(warp.lane_mask_.active_count == 1);

    // WarpIdentityPod mirrored via set_physical_* (WarpIdentityPod will
    // be populated by A4b's set_physical_* migration; for now it stays
    // default). Verify set_physical_warp_id still works.
    REQUIRE(warp.get_physical_warp_id() == 7);
    REQUIRE(warp.get_physical_block_id() == 2);
}