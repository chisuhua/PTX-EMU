// Unit tests for T2-3 POD split — warp_identity.h
// Verifies that WarpIdentityPod aggregates per-warp identity fields
// without behavior.

#include "ptxsim/contexts/warp_identity.h"
#include <catch_amalgamated.hpp>

using ptxsim::contexts::WarpIdentityPod;

TEST_CASE("WarpIdentityPod: default values are zero",
          "[contexts][pod][warp_identity]") {
    WarpIdentityPod pod;
    REQUIRE(pod.warp_id == 0);
    REQUIRE(pod.physical_warp_id == 0);
    REQUIRE(pod.physical_block_id == 0);
    REQUIRE(pod.pc == 0);
}

TEST_CASE("WarpIdentityPod: identity fields are assignable",
          "[contexts][pod][warp_identity]") {
    WarpIdentityPod pod;
    pod.warp_id = 1;
    pod.physical_warp_id = 8;
    pod.physical_block_id = 2;
    pod.pc = 42;

    REQUIRE(pod.warp_id == 1);
    REQUIRE(pod.physical_warp_id == 8);
    REQUIRE(pod.physical_block_id == 2);
    REQUIRE(pod.pc == 42);
}