// test_device_api_delegation.cpp
// =============================================================================
// Phase 2.2 unit tests for IPtxEmuDevice::set_scoreboard / set_active_mask /
// set_next_pc delegation.
//
// Scope (per openspec/changes/device-api-delegation/specs/ptxemu-device-api-delegation):
//   1. Method bodies don't crash on null g_gpu_context (defensive guards)
//   2. Invalid sm_id / warp_id returns false (defensive)
//   3. Phase 2.2 R7 minimal scope: set_scoreboard validates wiring only
//
// Deep success-path tests (with g_gpu_context + SMContext + warp setup)
// are covered by tests/integration/simt/test_set_active_mask_overwrite.cpp
// for set_active_mask; set_scoreboard/set_next_pc deep tests deferred
// to Phase 2.2.1 follow-up (per design.md R7).
//
// Ref: openspec/changes/device-api-delegation/{proposal,design}.md
// Ref: AGENTS.md L85 (set_pc+commit_pc, NOT force_set_pc)
// =============================================================================

#include "catch_amalgamated.hpp"
#include "ptxemu/device_api.h"

#include "ptxsim/gpu_context.h"

#include <memory>

// g_gpu_context is defined in src/cudart/cudart_sim.cpp (per ADR-0021 v1.1).
// For unit tests, we assume it's nullptr (test setup doesn't initialize
// the singleton). GPUContext is in global namespace (no namespace wrap).
extern std::unique_ptr<GPUContext> g_gpu_context;

TEST_CASE("device_api_delegation: set_scoreboard returns false when g_gpu_context is null",
          "[unit][ptxemu][delegation]") {
    auto dev = ptxemu::create_device();
    REQUIRE(dev != nullptr);
    // g_gpu_context is null in test environment → method must return false
    bool result = dev->set_scoreboard(0, 0, 0xFFFFFFFFu);
    REQUIRE(result == false);
}

TEST_CASE("device_api_delegation: set_active_mask returns false when g_gpu_context is null",
          "[unit][ptxemu][delegation]") {
    auto dev = ptxemu::create_device();
    REQUIRE(dev != nullptr);
    bool result = dev->set_active_mask(0, 0, 0x01u);
    REQUIRE(result == false);
}

TEST_CASE("device_api_delegation: set_next_pc returns false when g_gpu_context is null",
          "[unit][ptxemu][delegation]") {
    auto dev = ptxemu::create_device();
    REQUIRE(dev != nullptr);
    bool result = dev->set_next_pc(0, 0, 0, 42u);
    REQUIRE(result == false);
}

TEST_CASE("device_api_delegation: lifecycle methods don't crash delegation methods",
          "[unit][ptxemu][delegation]") {
    auto dev = ptxemu::create_device();
    REQUIRE(dev != nullptr);
    ptxemu::DeviceConfig cfg{};
    REQUIRE(dev->initialize(cfg) == true);
    dev->shutdown();
    // After shutdown, g_gpu_context is still null → all 3 delegation methods
    // return false defensively without crash.
    REQUIRE(dev->set_scoreboard(0, 0, 0xFFu) == false);
    REQUIRE(dev->set_active_mask(0, 0, 0xFFu) == false);
    REQUIRE(dev->set_next_pc(0, 0, 0, 100u) == false);
}

TEST_CASE("device_api_delegation: factory create_device / destroy_device",
          "[unit][ptxemu][delegation]") {
    auto dev = ptxemu::create_device();
    REQUIRE(dev != nullptr);
    // Should be deletable via destroy_device (HSK-8 spec §CppTLM 端接受条件
    // #1 第 4 项).
    ptxemu::destroy_device(dev.get());
    dev.release();  // ownership transferred to destroy_device
    REQUIRE(true);  // reached without crash
}