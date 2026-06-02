
#include "catch_amalgamated.hpp"
#include "ptxsim/thread_state.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/wbar.h"
#include "ptxsim/exec_mask.h"

#include <iostream>
#include <sstream>

using namespace ptxsim;

// ============================================================================
// Test Suite: Warp-Level Barrier Mechanism
// ============================================================================
// Tests for PTX ISA v6.0+ bar.warp.sync and activemask instructions
// ============================================================================

TEST_CASE("Wbar convergence barrier operations", "[simt][wbar][barrier]") {
    Wbar barrier;
    
    SECTION("Initialization on first arrive") {
        REQUIRE(barrier.is_initialized == false);
        
        // First thread arrives and initializes
        barrier.participation_mask = 0x0000000F;  // Lanes 0-3
        barrier.reconvergence_pc = 100;
        barrier.is_initialized = true;
        
        REQUIRE(barrier.is_initialized == true);
        REQUIRE(barrier.participation_mask == 0x0000000F);
        REQUIRE(barrier.reconvergence_pc == 100);
    }
    
    SECTION("Multiple threads arriving") {
        barrier.init(0x0000000F, 100);
        
        // Thread 0 arrives
        barrier.arrive(0);
        REQUIRE(barrier.count_arrived() == 1);
        REQUIRE(barrier.is_complete() == false);
        
        // Thread 1 arrives
        barrier.arrive(1);
        REQUIRE(barrier.count_arrived() == 2);
        REQUIRE(barrier.is_complete() == false);
        
        // Thread 2 arrives
        barrier.arrive(2);
        REQUIRE(barrier.count_arrived() == 3);
        REQUIRE(barrier.is_complete() == false);
        
        // Thread 3 arrives (last participant)
        barrier.arrive(3);
        REQUIRE(barrier.count_arrived() == 4);
        REQUIRE(barrier.is_complete() == true);
    }
    
    SECTION("Non-participant arrival") {
        barrier.init(0x0000000F, 100);  // Only lanes 0-3 participate
        
        // Lane 4 arrives (not a participant)
        barrier.arrive(4);
        REQUIRE(barrier.is_complete() == false);  // Should not complete
        
        // Actual participants arrive
        for (int i = 0; i < 4; ++i) {
            barrier.arrive(i);
        }
        REQUIRE(barrier.is_complete() == true);
    }
    
    SECTION("Reset after completion") {
        barrier.init(0x0000000F, 100);
        for (int i = 0; i < 4; ++i) {
            barrier.arrive(i);
        }
        REQUIRE(barrier.is_complete() == true);
        
        // Reset for next use
        barrier.reset();
        REQUIRE(barrier.is_initialized == false);
        REQUIRE(barrier.participation_mask == 0);
        REQUIRE(barrier.arrived_mask == 0);
    }
}

TEST_CASE("Warp-level barrier scenario (simulated)", "[simt][barrier][spinlock]") {
    WarpState warp;
    
    SECTION("Bar.warp.sync with divergence") {
        // Scenario: Lane 0 in spinlock, lanes 1-3 at barrier
        
        // Initialize exec mask (all active initially)
        warp.exec_mask = 0xFFFFFFFF;
        
        // Set up divergent execution
        warp.threads[0].pc = 10;  // In spinlock
        warp.threads[0].is_blocked = false;
        
        for (int i = 1; i < 32; ++i) {
            warp.threads[i].pc = 20;  // At barrier instruction
            warp.threads[i].is_blocked = false;
        }
        
        // Simulate bar.warp.sync execution
        // All threads participate in barrier
        auto& wbar = warp.wbars[0];
        wbar.participation_mask = warp.exec_mask;
        wbar.reconvergence_pc = 30;
        wbar.is_initialized = true;
        
        // Lane 0 arrives (still in spinlock initially)
        wbar.arrive(0);
        REQUIRE(wbar.count_arrived() == 1);
        
        // Simulate lane 0 exiting spinlock due to per-thread PC
        warp.threads[0].pc = 20;  // Now at barrier
        
        // Lane 0 arrives at barrier
        wbar.arrive(0);  // Already arrived, but simulates the flow
        
        // Other lanes arrive
        for (int i = 1; i < 32; ++i) {
            wbar.arrive(i);
        }
        
        // Barrier should be complete
        REQUIRE(wbar.is_complete() == true);
        
        // Release - set all to reconvergence PC
        for (int i = 0; i < 32; ++i) {
            if (wbar.participation_mask & (1u << i)) {
                warp.threads[i].pc = wbar.reconvergence_pc;
                warp.threads[i].is_blocked = false;
            }
        }
        
        // Verify all converged
        for (int i = 0; i < 32; ++i) {
            REQUIRE(warp.threads[i].pc == 30);
        }
    }
    
    SECTION("Selective participation") {
        // Only lanes 0-7 participate in barrier
        uint32_t participant_mask = 0x000000FF;
        
        warp.exec_mask = 0xFFFFFFFF;  // All lanes active
        
        // Set up barrier with selective participation
        auto& wbar = warp.wbars[0];
        wbar.participation_mask = participant_mask;
        wbar.reconvergence_pc = 50;
        wbar.is_initialized = true;
        wbar.arrived_mask = 0;
        
        // Only participating lanes arrive
        for (int i = 0; i < 8; ++i) {
            wbar.arrive(i);
        }
        
        // Non-participants (lanes 8-31) don't arrive
        // Barrier should still complete
        REQUIRE(wbar.count_participants() == 8);
        REQUIRE(wbar.count_arrived() == 8);
        REQUIRE(wbar.is_complete() == true);
    }
}

TEST_CASE("ExecMask for activemask instruction", "[simt][exec_mask][activemask]") {
    ExecMask mask;
    
    SECTION("Default full mask") {
        REQUIRE(mask.is_full() == true);
        REQUIRE(mask.value() == 0xFFFFFFFF);
        REQUIRE(mask.count_active_lanes() == 32);
    }
    
    SECTION("Partial mask (divergence)") {
        // Simulate exec mask after if-else divergence
        ExecMask partial(0x0000FFFF);  // Lanes 0-15 active
        
        REQUIRE(partial.count_active_lanes() == 16);
        REQUIRE(partial.is_full() == false);
        REQUIRE(partial.is_lane_active(0) == true);
        REQUIRE(partial.is_lane_active(15) == true);
        REQUIRE(partial.is_lane_active(16) == false);
    }
    
    SECTION("Active mask computation") {
        // Only even lanes active
        uint32_t even_mask = 0x55555555;  // 01010101...
        ExecMask even(even_mask);
        
        REQUIRE(even.count_active_lanes() == 16);
        REQUIRE(even.is_lane_active(0) == true);
        REQUIRE(even.is_lane_active(1) == false);
        REQUIRE(even.is_lane_active(2) == true);
        
        // Iterate over active lanes
        int lane = even.first_active_lane();
        int count = 0;
        while (lane >= 0) {
            REQUIRE((lane % 2) == 0);  // Should be even
            count++;
            lane = even.next_active_lane(lane);
        }
        REQUIRE(count == 16);
    }
    
    SECTION("Mask from warp state") {
        WarpState warp;
        warp.exec_mask = 0x0000000F;  // Lanes 0-3
        
        ExecMask mask(warp.exec_mask);
        REQUIRE(mask.count_active_lanes() == 4);
        
        // Simulate lane 0 exiting
        warp.threads[0].is_exited = true;
        // In real implementation, would update exec_mask
    }
}

TEST_CASE("Multiple barrier registers", "[simt][wbar][multi]") {
    WarpState warp;
    
    SECTION("Four warp barrier registers") {
        // Verify we have 4 barrier registers (hardware-typical)
        REQUIRE(warp.wbars.size() == 4);
        
        // Use different barrier registers
        warp.wbars[0].init(0x0000000F, 100);
        warp.wbars[1].init(0x000000F0, 200);
        warp.wbars[2].init(0x00000F00, 300);
        warp.wbars[3].init(0x0000F000, 400);
        
        // Each barrier operates independently
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int lane = i * 4 + j;
                warp.wbars[i].arrive(lane);
            }
            REQUIRE(warp.wbars[i].is_complete() == true);
        }
    }
    
    SECTION("Barrier nesting simulation") {
        // Nested barriers: outer barrier at PC 100, inner at PC 50
        auto& outer = warp.wbars[0];
        auto& inner = warp.wbars[1];
        
        outer.init(0xFFFFFFFF, 200);  // All lanes
        inner.init(0x0000000F, 100);  // Lanes 0-3 only
        
        // All lanes arrive at outer
        for (int i = 0; i < 32; ++i) {
            outer.arrive(i);
        }
        REQUIRE(outer.is_complete() == true);
        
        // Subset participates in inner
        for (int i = 0; i < 4; ++i) {
            inner.arrive(i);
        }
        REQUIRE(inner.is_complete() == true);
        
        // Verify independence
        REQUIRE(outer.reconvergence_pc == 200);
        REQUIRE(inner.reconvergence_pc == 100);
    }
}

TEST_CASE("Barrier completion helper functions", "[simt][wbar][utils]") {
    Wbar barrier;
    
    SECTION("Count operations") {
        barrier.init(0x0FFFFFFF, 100);  // Lanes 0-27
        
        REQUIRE(barrier.count_participants() == 28);
        REQUIRE(barrier.count_arrived() == 0);
        
        for (int i = 0; i < 28; ++i) {
            barrier.arrive(i);
        }
        
        REQUIRE(barrier.count_arrived() == 28);
    }
    
    SECTION("Is complete edge cases") {
        // Empty participation (invalid but should handle)
        barrier.participation_mask = 0;
        barrier.is_initialized = true;
        REQUIRE(barrier.is_complete() == false);
        
        // Single participant
        barrier.participation_mask = 0x00000001;
        barrier.arrived_mask = 0;
        REQUIRE(barrier.is_complete() == false);
        
        barrier.arrive(0);
        REQUIRE(barrier.is_complete() == true);
        
        // All 32 lanes
        barrier.reset();
        barrier.init(0xFFFFFFFF, 100);
        for (int i = 0; i < 32; ++i) {
            barrier.arrive(i);
        }
        REQUIRE(barrier.is_complete() == true);
    }
}

// ============================================================================
// Additional Tests: Spinlock & Advanced Wbar Scenarios
// ============================================================================

TEST_CASE("Wbar with atomicCAS spinlock simulation", "[simt][wbar][spinlock][atomic]") {
    // Simulates the classic spinlock deadlock scenario and its resolution
    
    SECTION("Simulated spinlock with atomicCAS") {
        WarpState warp;
        volatile int* lock = new int(0);  // Shared lock
        
        // Initial state: all threads at same PC
        for (int i = 0; i < 32; ++i) {
            warp.threads[i].pc = 10;
        }
        
        // Thread 0 attempts atomicCAS (spinlock owner)
        warp.threads[0].pc = 10;  // atomicCAS instruction
        warp.threads[0].is_blocked = false;
        
        // Threads 1-31 pass CAS check, reach barrier
        for (int i = 1; i < 32; ++i) {
            warp.threads[i].pc = 15;  // Barrier instruction
            warp.threads[i].is_blocked = true;
            warp.threads[i].status = ThreadStatus::Blocked;
        }
        
        // Verify only thread 0 is schedulable
        REQUIRE(warp.count_schedulable_lanes() == 1);
        REQUIRE(warp.threads[0].is_schedulable() == true);
        REQUIRE(warp.threads[1].is_schedulable() == false);
        
        // Simulate thread 0 spinning (CAS fails, loops back)
        warp.threads[0].pc = 10;
        REQUIRE(warp.threads[0].pc == 10);
        
        // Eventually CAS succeeds, thread 0 reaches barrier
        warp.threads[0].pc = 15;
        warp.threads[0].is_blocked = true;
        warp.threads[0].status = ThreadStatus::Blocked;
        
        // Now all threads at barrier - ready to release
        REQUIRE(warp.count_schedulable_lanes() == 0);
        
        delete lock;
    }
}

TEST_CASE("Wbar nested barrier scenario", "[simt][wbar][nested]") {
    WarpState warp;
    
    SECTION("Two-level nested barriers") {
        // Outer barrier (all 32 threads participate)
        auto& outer = warp.wbars[0];
        outer.init(0xFFFFFFFF, 200);  // All threads, reconverge at PC=200
        
        // Inner barrier (only lanes 0-15 participate)
        auto& inner = warp.wbars[1];
        inner.init(0x0000FFFF, 100);  // Lanes 0-15, reconverge at PC=100
        
        // Lanes 0-15 arrive at inner barrier
        for (int i = 0; i < 16; ++i) {
            inner.arrive(i);
        }
        
        // Inner barrier should complete
        REQUIRE(inner.is_complete() == true);
        REQUIRE(outer.is_complete() == false);
        
        // All 32 lanes arrive at outer barrier
        for (int i = 0; i < 32; ++i) {
            outer.arrive(i);
        }
        REQUIRE(outer.is_complete() == true);
    }
    
    SECTION("Sequential barrier reuse") {
        auto& barrier = warp.wbars[0];
        
        // First synchronization point
        barrier.init(0x0000000F, 100);
        for (int i = 0; i < 4; ++i) {
            barrier.arrive(i);
        }
        REQUIRE(barrier.is_complete() == true);
        
        // Reset and reuse for second synchronization
        barrier.reset();
        barrier.init(0x000000FF, 200);  // Different participants, different PC
        REQUIRE(barrier.count_participants() == 8);
        REQUIRE(barrier.reconvergence_pc == 200);
        
        for (int i = 0; i < 8; ++i) {
            barrier.arrive(i);
        }
        REQUIRE(barrier.is_complete() == true);
    }
}

TEST_CASE("Wbar with divergent control flow", "[simt][wbar][divergence]") {
    WarpState warp;
    
    SECTION("If-else-endif with barrier") {
        // Simulate: if (lane < 16) { ... } else { ... } __sync;
        auto& barrier = warp.wbars[0];
        
        // Initialize for full warp synchronization
        barrier.init(0xFFFFFFFF, 50);
        
        // Lanes 0-15 take if-path (PC advances through if-body)
        for (int i = 0; i < 16; ++i) {
            warp.threads[i].pc = 20;  // If-body
            warp.threads[i].is_active = true;
        }
        
        // Lanes 16-31 take else-path
        for (int i = 0; i < 32; ++i) {
            warp.threads[i].pc = 35;  // Else-body
            warp.threads[i].is_active = true;
        }
        
        // All converge at barrier
        for (int i = 0; i < 32; ++i) {
            barrier.arrive(i);
        }
        
        REQUIRE(barrier.is_complete() == true);
    }
    
    SECTION("Multiple divergence points") {
        // Thread state after multiple divergent branches
        for (int i = 0; i < 32; ++i) {
            warp.threads[i].pc = i % 10;  // Different PCs from divergence
            warp.threads[i].is_active = true;
        }
        
        // All threads must synchronize before next instruction
        auto& barrier = warp.wbars[0];
        barrier.init(0xFFFFFFFF, 100);
        
        for (int i = 0; i < 32; ++i) {
            barrier.arrive(i);
        }
        
        REQUIRE(barrier.is_complete() == true);
    }
}

TEST_CASE("Wbar partial participation scenarios", "[simt][wbar][partial]") {
    WarpState warp;
    
    SECTION("Power-of-2 participant counts") {
        // Test different power-of-2 participations (common in reductions)
        
        // 2 participants
        auto& b2 = warp.wbars[0];
        b2.init(0x00000003, 10);
        for (int i = 0; i < 2; ++i) {
            b2.arrive(i);
        }
        REQUIRE(b2.is_complete() == true);
        b2.reset();
        
        // 4 participants
        auto& b4 = warp.wbars[1];
        b4.init(0x0000000F, 20);
        for (int i = 0; i < 4; ++i) {
            b4.arrive(i);
        }
        REQUIRE(b4.is_complete() == true);
        b4.reset();
        
        // 8 participants
        auto& b8 = warp.wbars[2];
        b8.init(0x000000FF, 30);
        for (int i = 0; i < 8; ++i) {
            b8.arrive(i);
        }
        REQUIRE(b8.is_complete() == true);
    }
    
    SECTION("Non-contiguous participants") {
        // Even lanes only (lane 0, 2, 4, ..., 30)
        uint32_t even_mask = 0x55555555;  // 01010101...
        auto& barrier = warp.wbars[0];
        barrier.init(even_mask, 100);
        
        for (int i = 0; i < 32; i += 2) {
            barrier.arrive(i);
        }
        
        REQUIRE(barrier.count_participants() == 16);
        REQUIRE(barrier.is_complete() == true);
        
        // Odd lanes only (lane 1, 3, 5, ..., 31)
        barrier.reset();
        uint32_t odd_mask = 0xAAAAAAAA;  // 10101010...
        barrier.init(odd_mask, 200);
        
        for (int i = 1; i < 32; i += 2) {
            barrier.arrive(i);
        }
        
        REQUIRE(barrier.count_participants() == 16);
        REQUIRE(barrier.is_complete() == true);
    }
}

TEST_CASE("Wbar stress test", "[simt][wbar][stress]") {
    WarpState warp;
    
    SECTION("Rapid successive barriers") {
        // Simulate many quick barrier synchronizations
        const int num_sync = 100;
        
        for (int sync = 0; sync < num_sync; ++sync) {
            auto& barrier = warp.wbars[sync % 4];
            barrier.init(0xFFFFFFFF, sync);
            
            for (int i = 0; i < 32; ++i) {
                barrier.arrive(i);
            }
            
            REQUIRE(barrier.is_complete() == true);
            barrier.reset();
        }
        
        // All barriers completed successfully
        REQUIRE(true);
    }
    
    SECTION("All lanes arrive in random order") {
        auto& barrier = warp.wbars[0];
        barrier.init(0xFFFFFFFF, 100);
        
        // Arrive in non-sequential order
        std::vector<int> arrival_order;
        for (int i = 0; i < 32; ++i) {
            arrival_order.push_back(i);
        }
        
        // Reverse order
        for (int i = 31; i >= 0; --i) {
            barrier.arrive(arrival_order[i]);
            if (i > 0) {
                REQUIRE(barrier.is_complete() == false);
            }
        }
        REQUIRE(barrier.is_complete() == true);
    }
}

TEST_CASE("Wbar thread state transitions", "[simt][wbar][state]") {
    WarpState warp;
    
    SECTION("Active -> Blocked -> Active transition") {
        auto& barrier = warp.wbars[0];
        
        // Thread 0 waiting at barrier
        warp.threads[0].pc = 20;
        warp.threads[0].is_blocked = true;
        warp.threads[0].status = ThreadStatus::Blocked;
        
        // Other threads arrive
        barrier.init(0xFFFFFFFF, 50);
        for (int i = 0; i < 32; ++i) {
            barrier.arrive(i);
        }
        
        // Barrier complete - release all threads
        REQUIRE(barrier.is_complete() == true);
        
        // All threads become schedulable again
        for (int i = 0; i < 32; ++i) {
            warp.threads[i].pc = 50;          // Reconvergence PC
            warp.threads[i].is_blocked = false;
            warp.threads[i].status = ThreadStatus::Active;
        }
        
        REQUIRE(warp.count_schedulable_lanes() == 32);
    }
}

TEST_CASE("Wbar reconvergence PC verification", "[simt][wbar][pc]") {
    WarpState warp;
    
    SECTION("Different reconvergence PCs for different barriers") {
        // Each barrier has its own reconvergence point
        auto& b1 = warp.wbars[0];
        b1.init(0x000000FF, 100);  // 8 threads, PC=100
        
        auto& b2 = warp.wbars[1];
        b2.init(0x0000FF00, 200);  // 8 threads, PC=200
        
        auto& b3 = warp.wbars[2];
        b3.init(0x00FF0000, 300);  // 8 threads, PC=300
        
        auto& b4 = warp.wbars[3];
        b4.init(0xFF000000, 400);  // 8 threads, PC=400
        
        // Verify each barrier has correct reconvergence PC
        for (int i = 0; i < 8; ++i) {
            b1.arrive(i);
            b2.arrive(i + 8);
            b3.arrive(i + 16);
            b4.arrive(i + 24);
        }
        
        REQUIRE(b1.is_complete() == true);
        REQUIRE(b2.is_complete() == true);
        REQUIRE(b3.is_complete() == true);
        REQUIRE(b4.is_complete() == true);
        
        // Verify each barrier's reconvergence PC
        REQUIRE(warp.wbars[0].reconvergence_pc == 100);
        REQUIRE(warp.wbars[1].reconvergence_pc == 200);
        REQUIRE(warp.wbars[2].reconvergence_pc == 300);
        REQUIRE(warp.wbars[3].reconvergence_pc == 400);
    }
}

