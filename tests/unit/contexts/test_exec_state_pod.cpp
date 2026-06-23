// Unit tests for T2-3 POD split — exec_state.h
// Verifies that ExecStatePod aggregates per-thread identity and state-machine
// fields without behavior, matching the design intent of the god-class split.

#include "ptxsim/contexts/exec_state.h"
#include <catch_amalgamated.hpp>

using ptxsim::contexts::ExecStatePod;

TEST_CASE("ExecStatePod: default constructor zero-initializes identity", "[contexts][pod][exec_state]") {
    ExecStatePod pod;
    REQUIRE(pod.BlockIdx.x == 0);
    REQUIRE(pod.BlockIdx.y == 0);
    REQUIRE(pod.BlockIdx.z == 0);
    REQUIRE(pod.ThreadIdx.x == 0);
    REQUIRE(pod.ThreadIdx.y == 0);
    REQUIRE(pod.ThreadIdx.z == 0);
    REQUIRE(pod.warp_id_ == 0);
    REQUIRE(pod.lane_id_ == 0);
}

TEST_CASE("ExecStatePod: default GridDim is (1,1,1)", "[contexts][pod][exec_state]") {
    ExecStatePod pod;
    REQUIRE(pod.GridDim.x == 1);
    REQUIRE(pod.GridDim.y == 1);
    REQUIRE(pod.GridDim.z == 1);
}

TEST_CASE("ExecStatePod: default BlockDim is (1,1,1)", "[contexts][pod][exec_state]") {
    ExecStatePod pod;
    REQUIRE(pod.BlockDim.x == 1);
    REQUIRE(pod.BlockDim.y == 1);
    REQUIRE(pod.BlockDim.z == 1);
}

TEST_CASE("ExecStatePod: default state is IDLE", "[contexts][pod][exec_state]") {
    ExecStatePod pod;
    REQUIRE(pod.state == IDLE);
}

TEST_CASE("ExecStatePod: default bar_id is 0", "[contexts][pod][exec_state]") {
    ExecStatePod pod;
    REQUIRE(pod.bar_id == 0);
}

TEST_CASE("ExecStatePod: assignment and direct field write are POD-like",
          "[contexts][pod][exec_state]") {
    ExecStatePod pod;
    pod.warp_id_ = 5;
    pod.lane_id_ = 17;
    pod.ThreadIdx = Dim3{42, 0, 0};
    pod.state = RUN;
    pod.bar_id = 3;

    REQUIRE(pod.warp_id_ == 5);
    REQUIRE(pod.lane_id_ == 17);
    REQUIRE(pod.ThreadIdx.x == 42);
    REQUIRE(pod.state == RUN);
    REQUIRE(pod.bar_id == 3);
}