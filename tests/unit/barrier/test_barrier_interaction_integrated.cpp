#include "catch_amalgamated.hpp"
#include "ptxsim/barrier/barrier_module.h"

using ptxsim::BarrierModule;

TEST_CASE("BarrierModule interaction (Phase 7 Wbar replacement)", "[simt][interaction]") {
    BarrierModule bm;
    
    SECTION("Warp barrier init and arrive") {
        bm.init_warp_barrier(0, 0x0000FFFFu, 10, 5);
        auto* wb = bm.get_warp_barrier(0);
        REQUIRE(wb->is_initialized());
        REQUIRE(wb->get_expected_count() == 16);
        
        for (int i = 0; i < 16; i++) wb->arrive(i);
        REQUIRE(wb->is_complete());
    }
    
    SECTION("Warp barrier reset") {
        bm.init_warp_barrier(0, 0x000000FFu, 20, 5);
        auto* wb = bm.get_warp_barrier(0);
        for (int i = 0; i < 8; i++) wb->arrive(i);
        REQUIRE(wb->is_complete());
        bm.get_warp_barrier(0)->reset();
        REQUIRE(!wb->is_initialized());
    }
}
