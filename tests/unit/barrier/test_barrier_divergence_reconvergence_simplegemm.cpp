#include "catch_amalgamated.hpp"
#include "ptxsim/barrier/barrier_module.h"
#include "ptxsim/warp_state.h"

using ptxsim::BarrierModule;
using ptxsim::WarpState;
using ptxsim::ThreadStatus;

// Legacy Wbar tests removed in Phase 7 (migrate-bar-warp-sync-to-barrier-module).
// Equivalent tests exist in unit_barrier_module and integration_post_barrier_*.

TEST_CASE("BarrierModule re-init preserves arrived_mask (BUG-RECONVERGENCE-SIMPLEGEMM)", "[barrier][regression]") {
    BarrierModule bm;
    // First half arrives
    bm.init_warp_barrier(0, 0xFFFFFFFFu, 70, 50);
    for (int i = 16; i < 32; i++) bm.get_warp_barrier(0)->arrive(i);
    REQUIRE(bm.get_warp_barrier(0)->get_arrived_count() == 16);
    
    // Re-init (simulates force_reconvergence second half)
    bm.init_warp_barrier(0, 0xFFFFFFFFu, 70, 50);
    // First half's arrivals preserved via is_initialized_ branch
    REQUIRE(bm.get_warp_barrier(0)->get_arrived_count() == 16);
    REQUIRE(!bm.is_warp_barrier_complete(0));
    
    // Second half arrives
    for (int i = 0; i < 16; i++) bm.get_warp_barrier(0)->arrive(i);
    REQUIRE(bm.is_warp_barrier_complete(0));
}

TEST_CASE("BarrierModule post-completion release", "[barrier][regression]") {
    BarrierModule bm;
    bm.init_warp_barrier(0, 0xFFFFFFFFu, 100, 0);
    for (int i = 0; i < 32; i++) bm.get_warp_barrier(0)->arrive(i);
    REQUIRE(bm.is_warp_barrier_complete(0));
    // Release resets state
    bm.get_warp_barrier(0)->reset();
    REQUIRE(!bm.get_warp_barrier(0)->is_initialized());
}
