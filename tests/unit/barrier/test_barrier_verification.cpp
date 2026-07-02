#include "catch_amalgamated.hpp"
#include "ptxsim/barrier/barrier_module.h"

// Legacy Wbar verification tests removed in Phase 7.
// Equivalent WarpBarrier tests exist in unit_barrier_module.

using namespace ptxsim;

TEST_CASE("BarrierModule basic init", "[barrier][verification]") {
    BarrierModule bm;
    auto* wb = bm.get_warp_barrier(0);
    REQUIRE(!wb->is_initialized());
    
    bm.init_warp_barrier(0, 0x0000000F, 100, 10);
    REQUIRE(wb->is_initialized());
}