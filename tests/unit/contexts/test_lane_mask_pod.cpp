// Unit tests for T2-3 POD split — lane_mask.h
// Verifies that LaneMaskPod aggregates per-warp lane-activity tracking
// fields without behavior.

#include "ptxsim/contexts/lane_mask.h"
#include <catch_amalgamated.hpp>

using ptxsim::contexts::LaneMaskPod;

TEST_CASE("LaneMaskPod: default active_mask is all false",
          "[contexts][pod][lane_mask]") {
    LaneMaskPod pod;
    for (int i = 0; i < LaneMaskPod::WARP_SIZE; ++i) {
        REQUIRE(pod.active_mask[i] == false);
    }
}

TEST_CASE("LaneMaskPod: default warp_thread_ids is all -1 (uninitialized)",
          "[contexts][pod][lane_mask]") {
    LaneMaskPod pod;
    for (int i = 0; i < LaneMaskPod::WARP_SIZE; ++i) {
        REQUIRE(pod.warp_thread_ids[i] == 0);
    }
}

TEST_CASE("LaneMaskPod: default active_count is 0",
          "[contexts][pod][lane_mask]") {
    LaneMaskPod pod;
    REQUIRE(pod.active_count == 0);
}

TEST_CASE("LaneMaskPod: default divergence_detected is false",
          "[contexts][pod][lane_mask]") {
    LaneMaskPod pod;
    REQUIRE_FALSE(pod.divergence_detected);
}

TEST_CASE("LaneMaskPod: default is_scheduled_ is false",
          "[contexts][pod][lane_mask]") {
    LaneMaskPod pod;
    REQUIRE_FALSE(pod.is_scheduled_);
}

TEST_CASE("LaneMaskPod: per-lane fields can be written",
          "[contexts][pod][lane_mask]") {
    LaneMaskPod pod;
    pod.active_mask[5] = true;
    pod.warp_thread_ids[5] = 42;
    pod.active_count = 17;
    pod.divergence_detected = true;
    pod.is_scheduled_ = true;

    REQUIRE(pod.active_mask[5]);
    REQUIRE(pod.warp_thread_ids[5] == 42);
    REQUIRE(pod.active_count == 17);
    REQUIRE(pod.divergence_detected);
    REQUIRE(pod.is_scheduled_);
}