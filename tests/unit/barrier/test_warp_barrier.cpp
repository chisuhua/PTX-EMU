#include "catch_amalgamated.hpp"
#include "ptxsim/warp_state.h"
#include "ptxsim/exec_mask.h"

using namespace ptxsim;

// ============================================================================
// Test Suite: WarpState + ExecMask (Wbar legacy tests removed in Phase 7)
// ============================================================================
// Legacy Wbar struct and warp_state.wbars[] have been deleted as part of
// migrate-bar-warp-sync-to-barrier-module (Phase 7). Equivalent WarpBarrier
// tests exist in unit_barrier_module and integration_barrier tests.

TEST_CASE("ExecMask for activemask instruction", "[simt][exec_mask][activemask]") {
    ExecMask mask;
    
    SECTION("Default full mask") {
        REQUIRE(mask.is_full() == true);
        REQUIRE(mask.value() == 0xFFFFFFFF);
        REQUIRE(mask.count_active_lanes() == 32);
    }
    
    SECTION("Partial mask (divergence)") {
        ExecMask partial(0x0000FFFF);
        REQUIRE(partial.count_active_lanes() == 16);
        REQUIRE(partial.is_full() == false);
        REQUIRE(partial.is_lane_active(0) == true);
        REQUIRE(partial.is_lane_active(15) == true);
        REQUIRE(partial.is_lane_active(16) == false);
    }
    
    SECTION("Active mask computation") {
        uint32_t even_mask = 0x55555555;
        ExecMask even(even_mask);
        REQUIRE(even.count_active_lanes() == 16);
    }
    
    SECTION("Mask from warp state") {
        WarpState warp;
        warp.exec_mask = 0x0000000F;
        ExecMask mask(warp.exec_mask);
        REQUIRE(mask.count_active_lanes() == 4);
    }
}

TEST_CASE("WarpState thread state transitions", "[simt][state]") {
    WarpState warp;
    
    SECTION("Active -> Blocked -> Active transition") {
        warp.threads[0].pc = 20;
        warp.threads[0].is_blocked = true;
        warp.threads[0].status = ThreadStatus::Blocked;
        
        // Release all threads
        for (int i = 0; i < 32; ++i) {
            warp.threads[i].pc = 50;
            warp.threads[i].is_blocked = false;
            warp.threads[i].status = ThreadStatus::Active;
        }
        
        REQUIRE(warp.count_schedulable_lanes() == 32);
    }
}

TEST_CASE("WarpState atomicCAS spinlock simulation", "[simt][spinlock]") {
    SECTION("Simulated spinlock with atomicCAS") {
        WarpState warp;
        
        for (int i = 0; i < 32; ++i) {
            warp.threads[i].pc = 10;
        }
        
        warp.threads[0].is_blocked = false;
        for (int i = 1; i < 32; ++i) {
            warp.threads[i].is_blocked = true;
            warp.threads[i].status = ThreadStatus::Blocked;
        }
        
        REQUIRE(warp.count_schedulable_lanes() == 1);
        REQUIRE(warp.threads[0].is_schedulable() == true);
    }
}