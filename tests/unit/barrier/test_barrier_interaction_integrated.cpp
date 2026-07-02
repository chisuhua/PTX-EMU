/**
 * I1-I2: Thread scheduling order & barrier interaction tests.
 * Legacy Wbar tests removed in Phase 7 (migrate-bar-warp-sync-to-barrier-module).
 * Equivalent tests exist in unit_barrier_module and integration_barrier.
 */

#include "catch_amalgamated.hpp"
#include "ptxsim/warp_state.h"
#include <cstdint>

using ptxsim::WarpState;

TEST_CASE("WarpState basic scheduling invariants", "[simt][interaction]") {
    WarpState warp;
    
    SECTION("Default state has 32 threads") {
        REQUIRE(warp.count_active_lanes() == 0);
    }
    
    SECTION("Thread activation/deactivation") {
        warp.threads[0].is_active = true;
        warp.threads[31].is_active = true;
        REQUIRE(warp.count_active_lanes() == 2);
        
        warp.threads[0].is_active = false;
        REQUIRE(warp.count_active_lanes() == 1);
    }
}