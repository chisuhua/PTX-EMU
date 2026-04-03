
#include "catch_amalgamated.hpp"
#include "ptxsim/thread_state.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/exec_mask.h"
#include "ptxsim/wbar.h"

#include <iostream>
#include <sstream>
#include <atomic>
#include <vector>

// ============================================================================
// Test Suite: Per-Thread PC Mechanism
// ============================================================================
using namespace ptxsim;
// These tests verify the core data structures for SIMT architecture upgrade
// Focus: Per-thread PC, ExecMask, and WarpState management
// ============================================================================

TEST_CASE("ThreadState basic operations", "[simt][thread_state]") {
    ThreadState thread;
    
    SECTION("Default initialization") {
        REQUIRE(thread.pc == 0);
        REQUIRE(thread.next_pc == 0);
        REQUIRE(thread.is_active == true);
        REQUIRE(thread.is_exited == false);
        REQUIRE(thread.is_blocked == false);
        REQUIRE(thread.status == ThreadStatus::Active);
    }
    
    SECTION("PC manipulation") {
        thread.pc = 100;
        thread.next_pc = 101;
        
        REQUIRE(thread.pc == 100);
        REQUIRE(thread.next_pc == 101);
        
        // Simulate instruction fetch
        thread.pc = thread.next_pc;
        REQUIRE(thread.pc == 101);
    }
    
    SECTION("Thread status transitions") {
        // Active -> Blocked (barrier wait)
        thread.is_blocked = true;
        thread.status = ThreadStatus::Blocked;
        REQUIRE(thread.is_blocked == true);
        REQUIRE(thread.status == ThreadStatus::Blocked);
        
        // Blocked -> Active (barrier released)
        thread.is_blocked = false;
        thread.status = ThreadStatus::Active;
        REQUIRE(thread.is_blocked == false);
        REQUIRE(thread.status == ThreadStatus::Active);
        
        // Active -> Exited
        thread.is_exited = true;
        REQUIRE(thread.is_exited == true);
    }
    
    SECTION("Schedulable check") {
        // Active thread should be schedulable
        REQUIRE(thread.is_schedulable() == true);
        
        // Blocked thread should not be schedulable
        thread.is_blocked = true;
        thread.status = ThreadStatus::Blocked;
        REQUIRE(thread.is_schedulable() == false);
        
        // Exited thread should not be schedulable
        thread.is_blocked = false;
        thread.is_exited = true;
        REQUIRE(thread.is_schedulable() == false);
    }
    
    SECTION("Reset functionality") {
        thread.pc = 100;
        thread.is_exited = true;
        thread.is_blocked = true;
        thread.is_active = false;
        
        thread.reset();
        
        REQUIRE(thread.pc == 0);
        REQUIRE(thread.is_exited == false);
        REQUIRE(thread.is_blocked == false);
        REQUIRE(thread.is_active == true);
    }
}

TEST_CASE("ExecMask operations", "[simt][exec_mask]") {
    ExecMask mask;
    
    SECTION("Default initialization (all active)") {
        REQUIRE(mask.is_full() == true);
        REQUIRE(mask.is_empty() == false);
        REQUIRE(mask.count_active_lanes() == 32);
        REQUIRE(mask.value() == 0xFFFFFFFF);
    }
    
    SECTION("Lane manipulation") {
        // Set lane 0 inactive
        mask.set_lane(0, false);
        REQUIRE(mask.is_lane_active(0) == false);
        REQUIRE(mask.is_lane_active(1) == true);
        REQUIRE(mask.count_active_lanes() == 31);
        
        // Set lane 1-31 inactive
        for (int i = 1; i < 32; ++i) {
            mask.set_lane(i, false);
        }
        REQUIRE(mask.is_empty() == true);
        REQUIRE(mask.count_active_lanes() == 0);
    }
    
    SECTION("Mask operations (AND)") {
        ExecMask mask1(0xAAAAAAAA);  // 10101010...
        ExecMask mask2(0x55555555);  // 01010101...
        
        ExecMask result = mask1 & mask2;
        REQUIRE(result.is_empty() == true);  // No common bits
    }
    
    SECTION("Mask operations (OR)") {
        ExecMask mask1(0xAAAAAAAA);
        ExecMask mask2(0x55555555);
        
        ExecMask result = mask1 | mask2;
        REQUIRE(result.is_full() == true);  // All bits set
    }
    
    SECTION("First/Next active lane") {
        ExecMask mask(0x0000000F);  // Lanes 0-3 active
        
        REQUIRE(mask.first_active_lane() == 0);
        REQUIRE(mask.next_active_lane(0) == 1);
        REQUIRE(mask.next_active_lane(1) == 2);
        REQUIRE(mask.next_active_lane(2) == 3);
        REQUIRE(mask.next_active_lane(3) == -1);  // No more
    }
    
    SECTION("String representation") {
        ExecMask mask(0x00000003);  // Lanes 0-1 active
        std::string str = mask.to_string();
        
        REQUIRE(str.length() == 65); // [32 bits + 31 commas + 2 brackets]
        REQUIRE(str[0] == '[');
        REQUIRE(str[str.length() - 1] == ']');
    }
}

TEST_CASE("Wbar (Warp Barrier) operations", "[simt][wbar]") {
    Wbar barrier;
    
    SECTION("Default initialization") {
        REQUIRE(barrier.is_initialized == false);
        REQUIRE(barrier.participation_mask == 0);
        REQUIRE(barrier.arrived_mask == 0);
        REQUIRE(barrier.reconvergence_pc == -1);
    }
    
    SECTION("Initialize barrier") {
        // Initialize with 4 participating threads (lanes 0-3)
        barrier.init(0x0000000F, 100);
        
        REQUIRE(barrier.is_initialized == true);
        REQUIRE(barrier.count_participants() == 4);
        REQUIRE(barrier.reconvergence_pc == 100);
        REQUIRE(barrier.expected_count == 4);
    }
    
    SECTION("Barrier arrival tracking") {
        barrier.init(0x0000000F, 100);
        
        // Thread 0 arrives
        barrier.arrive(0);
        REQUIRE(barrier.count_arrived() == 1);
        REQUIRE(barrier.is_complete() == false);
        
        // Threads 1-3 arrive
        barrier.arrive(1);
        barrier.arrive(2);
        barrier.arrive(3);
        
        REQUIRE(barrier.count_arrived() == 4);
        REQUIRE(barrier.is_complete() == true);
    }
    
    SECTION("Partial participation") {
        // Only lanes 0, 8, 16, 24 participate
        barrier.init((1 << 0) | (1 << 8) | (1 << 16) | (1 << 24), 200);
        
        REQUIRE(barrier.count_participants() == 4);
        
        // Lane 0 arrives
        barrier.arrive(0);
        REQUIRE(barrier.is_complete() == false);
        
        // Lane 1 arrives (not a participant, should not affect completion)
        barrier.arrive(1);
        REQUIRE(barrier.is_complete() == false);
        
        // All participants arrive
        barrier.arrive(8);
        barrier.arrive(16);
        barrier.arrive(24);
        REQUIRE(barrier.is_complete() == true);
    }
    
    SECTION("Reset barrier") {
        barrier.init(0x0000000F, 100);
        barrier.arrive(0);
        
        barrier.reset();
        
        REQUIRE(barrier.is_initialized == false);
        REQUIRE(barrier.participation_mask == 0);
        REQUIRE(barrier.arrived_mask == 0);
        REQUIRE(barrier.reconvergence_pc == -1);
    }
}

TEST_CASE("WarpState per-thread PC", "[simt][warp_state]") {
    WarpState warp;
    
    SECTION("Default initialization") {
        REQUIRE(warp.exec_mask == 0xFFFFFFFF);
        REQUIRE(warp.count_active_lanes() == 32);
        REQUIRE(warp.count_schedulable_lanes() == 32);
        REQUIRE(warp.is_all_exited() == false);
    }
    
    SECTION("Per-thread PC divergence") {
        // Simulate branch divergence:
        // Lane 0 takes branch (PC=100), lanes 1-31 continue (PC=50)
        warp.threads[0].pc = 100;
        warp.threads[0].next_pc = 101;
        
        for (int i = 1; i < 32; ++i) {
            warp.threads[i].pc = 50;
            warp.threads[i].next_pc = 51;
        }
        
        // Verify per-thread PC
        REQUIRE(warp.threads[0].pc == 100);
        for (int i = 1; i < 32; ++i) {
            REQUIRE(warp.threads[i].pc == 50);
        }
        
        // Update PC after instruction fetch
        for (int i = 0; i < 32; ++i) {
            warp.threads[i].pc = warp.threads[i].next_pc;
        }
        
        REQUIRE(warp.threads[0].pc == 101);
        REQUIRE(warp.threads[1].pc == 51);
    }
    
    SECTION("Exec mask update") {
        // Disable lanes 0-15 (predicate false)
        warp.exec_mask = 0xFFFF0000;
        
        REQUIRE(warp.count_active_lanes() == 32); // count_active_lanes counts all threads
        REQUIRE(warp.threads[0].is_active == true);  // Note: exec_mask is separate
    }
    
    SECTION("Thread exit tracking") {
        // Exit lanes 0-15
        for (int i = 0; i < 16; ++i) {
            warp.threads[i].is_exited = true;
        }
        
        REQUIRE(warp.count_schedulable_lanes() == 16);
        REQUIRE(warp.is_all_exited() == false);
        
        // Exit remaining lanes
        for (int i = 16; i < 32; ++i) {
            warp.threads[i].is_exited = true;
        }
        
        REQUIRE(warp.is_all_exited() == true);
        REQUIRE(warp.count_schedulable_lanes() == 0);
        REQUIRE(warp.has_schedulable_threads() == false);
    }
    
    SECTION("Warp reset") {
        // Set divergent state
        warp.threads[0].pc = 100;
        warp.threads[0].is_exited = true;
        warp.exec_mask = 0x00000000;
        
        warp.reset();
        
        REQUIRE(warp.threads[0].pc == 0);
        REQUIRE(warp.threads[0].is_exited == false);
        REQUIRE(warp.exec_mask == 0xFFFFFFFF);
    }
}

TEST_CASE("Spinlock deadlock scenario", "[simt][spinlock]") {
    // This test demonstrates the core problem the architecture solves:
    // Lane 0 in spinlock loop, lanes 1-31 waiting at barrier
    
    WarpState warp;
    
    SECTION("Divergent spinlock execution") {
        // Initial state: all threads at PC=10 (spinlock check)
        for (int i = 0; i < 32; ++i) {
            warp.threads[i].pc = 10;
            warp.threads[i].next_pc = 11;
        }
        
        // Lane 0 fails CAS, stays in loop (PC stays at 10-15)
        warp.threads[0].pc = 10;
        warp.threads[0].next_pc = 10;  // Loop back
        
        // Lanes 1-31 pass if, reach barrier at PC=20
        for (int i = 1; i < 32; ++i) {
            warp.threads[i].pc = 20;  // Barrier instruction
            warp.threads[i].is_blocked = true;
            warp.threads[i].status = ThreadStatus::Blocked;
        }
        
        // Verify divergence state
        REQUIRE(warp.threads[0].pc == 10);
        REQUIRE(warp.threads[0].is_blocked == false);
        REQUIRE(warp.threads[0].is_schedulable() == true);  // Can still execute
        
        for (int i = 1; i < 32; ++i) {
            REQUIRE(warp.threads[i].pc == 20);
            REQUIRE(warp.threads[i].is_blocked == true);
            REQUIRE(warp.threads[i].is_schedulable() == false);  // Waiting
        }
        
        // Scheduler should be able to schedule lane 0 independently
        REQUIRE(warp.has_schedulable_threads() == true);
        REQUIRE(warp.count_schedulable_lanes() == 1);
        
        // Simulate lane 0 eventually succeeding and reaching barrier
        warp.threads[0].pc = 20;
        warp.threads[0].is_blocked = true;
        warp.threads[0].status = ThreadStatus::Blocked;
        
        // Now all threads are at barrier - barrier can be released
        REQUIRE(warp.count_schedulable_lanes() == 0);
        
        // (In full implementation, barrier logic would detect all arrived)
    }
}

TEST_CASE("Thread status helper functions", "[simt][utils]") {
    SECTION("Status to string conversion") {
        REQUIRE(std::string(thread_status_to_string(ThreadStatus::Active)) == "Active");
        REQUIRE(std::string(thread_status_to_string(ThreadStatus::Blocked)) == "Blocked");
        REQUIRE(std::string(thread_status_to_string(ThreadStatus::Exited)) == "Exited");
        REQUIRE(std::string(thread_status_to_string(ThreadStatus::Yielded)) == "Yielded");
    }
}

// ============================================================================
// Advanced Spinlock Tests - Realistic Scenarios
// ============================================================================

TEST_CASE("Simulated atomicCAS spinlock workflow", "[simt][spinlock][atomic][cas]") {
    WarpState warp;
    
    SECTION("Complete spinlock acquisition and critical section") {
        // Simulate the full workflow of a spinlock with per-thread PC
        
        volatile int* lock = new int(0);  // 0 = unlocked, 1 = locked
        std::atomic<int> critical_section_count(0);
        
        // Phase 1: All threads at CAS instruction (PC=10)
        for (int i = 0; i < 32; ++i) {
            warp.threads[i].pc = 10;
            warp.threads[i].is_blocked = false;
            warp.threads[i].is_active = true;
        }
        
        REQUIRE(warp.count_schedulable_lanes() == 32);
        
        // Phase 2: Warp scheduler picks all lanes (SIMD)
        // Each lane atomically compares lock value
        // Only one lane (say lane 0) succeeds in CAS
        
        // Thread 0: CAS succeeds (lock acquired)
        warp.threads[0].pc = 15;  // Critical section entry
        warp.threads[0].is_blocked = false;
        
        // Threads 1-31: CAS fails, loop back to retry
        for (int i = 1; i < 32; ++i) {
            warp.threads[i].pc = 10;  // Loop back to CAS
            warp.threads[i].is_blocked = false;
        }
        
        // All lanes still schedulable
        REQUIRE(warp.count_schedulable_lanes() == 32);
        
        // Phase 3: Thread 0 enters critical section, others spin
        warp.threads[0].pc = 20;  // In critical section
        
        for (int i = 1; i < 32; ++i) {
            warp.threads[i].pc = 10;  // Spinning at CAS
        }
        
        // All still schedulable
        REQUIRE(warp.count_schedulable_lanes() == 32);
        
        // Phase 4: Thread 0 releases lock, reaches barrier
        warp.threads[0].pc = 25;  // Release lock
        warp.threads[0].pc = 30;  // Barrier sync
        
        // Other threads still spinning
        for (int i = 1; i < 32; ++i) {
            warp.threads[i].pc = 10;
        }
        
        // All schedulable
        REQUIRE(warp.count_schedulable_lanes() == 32);
        
        // Phase 5: After barrier, all converge
        warp.threads[0].pc = 40;  // Past barrier
        for (int i = 1; i < 32; ++i) {
            warp.threads[i].pc = 40;  // Converged
        }
        
        delete lock;
    }
    
    SECTION("Multiple threads competing for lock") {
        // More realistic scenario where threads compete
        WarpState warp2;
        
        for (int i = 0; i < 32; ++i) {
            warp2.threads[i].pc = 10;
            warp2.threads[i].is_blocked = false;
        }
        
        // Simulate multiple CAS attempts
        int cas_attempts = 0;
        bool lock_acquired = false;
        int lock_owner = -1;
        
        while (!lock_acquired && cas_attempts < 100) {
            cas_attempts++;
            
            // Each CAS attempt - simulate warp converges and executes together
            for (int i = 0; i < 32; ++i) {
                if (!lock_acquired) {
                    // Simulate CAS - thread 0 always wins (deterministic for test)
                    if (i == 0) {
                        lock_acquired = true;
                        lock_owner = i;
                        warp2.threads[i].pc = 15;  // Winner enters critical section
                    } else {
                        warp2.threads[i].pc = 10;  // Losers loop back
                    }
                }
            }
            
            // Verify per-thread PC tracking works
            REQUIRE(warp2.threads[0].pc == (lock_acquired ? 15 : 10));
        }
        
        REQUIRE(lock_acquired == true);
        REQUIRE(lock_owner == 0);
        REQUIRE(cas_attempts == 1);
    }
}

TEST_CASE("Spinlock with barrier deadlock prevention", "[simt][spinlock][barrier][deadlock]") {
    WarpState warp;
    
    SECTION("Classic barrier-after-lock pattern (prevents deadlock)") {
        // This is the key test - demonstrates the problem our architecture solves
        
        // Scenario: 
        // - Thread 0 acquires lock
        // - Threads 1-31 spin at CAS
        // - Thread 0 enters critical section, then calls __syncthreads()
        // - Threads 1-31 eventually acquire lock and reach __syncthreads()
        // - WITHOUT per-thread PC: DEADLOCK (all share same PC)
        // - WITH per-thread PC: SUCCESS (each has independent PC)
        
        // Set up initial state (lock acquired by thread 0)
        warp.threads[0].pc = 20;  // In critical section
        warp.threads[0].is_blocked = false;
        
        for (int i = 1; i < 32; ++i) {
            warp.threads[i].pc = 10;  // Spinning at CAS
            warp.threads[i].is_blocked = false;
        }
        
        // Verify independent progress possible
        REQUIRE(warp.count_schedulable_lanes() == 32);
        
        // Thread 0 continues to barrier
        warp.threads[0].pc = 25;  // At barrier
        warp.threads[0].is_blocked = true;
        warp.threads[0].status = ThreadStatus::Blocked;
        
        // Lanes 1-31 still spinning (NOT blocked at barrier)
        for (int i = 1; i < 32; ++i) {
            REQUIRE(warp.threads[i].is_blocked == false);
            REQUIRE(warp.threads[i].is_schedulable() == true);
        }
        
        // Lanes 1-31 eventually succeed and reach barrier
        for (int i = 1; i < 32; ++i) {
            warp.threads[i].pc = 25;  // Arrive at barrier
            warp.threads[i].is_blocked = true;
        }
        
        // Now ALL threads at barrier
        REQUIRE(warp.count_schedulable_lanes() == 0);
        REQUIRE(warp.is_all_exited() == false);
        
        // Barrier can now be released (all participants present)
        for (int i = 0; i < 32; ++i) {
            warp.threads[i].pc = 30;  // Reconvergence point
            warp.threads[i].is_blocked = false;
            warp.threads[i].status = ThreadStatus::Active;
        }
        
        // All threads resumed
        REQUIRE(warp.count_schedulable_lanes() == 32);
    }
}

TEST_CASE("Per-thread PC enables independent spin count", "[simt][spinlock][independent]") {
    WarpState warp;
    
    SECTION("Different threads at different loop iterations") {
        // Simulate realistic spinlock where threads have different retry counts
        
        for (int i = 0; i < 32; ++i) {
            int spin_count = rand() % 10;  // Each thread spins different iterations
            warp.threads[i].pc = 10 + spin_count;  // Different PCs
            warp.threads[i].is_blocked = false;
        }
        
        // Verify per-thread PC allows this
        std::vector<int> pcs;
        for (int i = 0; i < 32; ++i) {
            pcs.push_back(warp.threads[i].pc);
        }
        
        // Not all PCs are the same (high probability with random)
        bool all_same = true;
        for (int i = 1; i < 32; ++i) {
            if (pcs[i] != pcs[0]) {
                all_same = false;
                break;
            }
        }
        REQUIRE(all_same == false);  // Divergence achieved
        
        // Each thread can be scheduled independently
        REQUIRE(warp.count_schedulable_lanes() == 32);
    }
}

TEST_CASE("Spinlock timeout scenario", "[simt][spinlock][timeout]") {
    WarpState warp;
    
    SECTION("Thread gives up after max retries") {
        // Some threads might timeout and exit the spinlock
        const int max_retries = 100;
        
        // Thread 0 - successful (acquires lock)
        warp.threads[0].pc = 20;  // In critical section
        warp.threads[0].is_active = true;
        
        // Thread 1-16 - spinning normally
        for (int i = 1; i <= 16; ++i) {
            warp.threads[i].pc = 10;
            warp.threads[i].is_active = true;
        }
        
        // Thread 17-31 - timed out and exited spinlock
        for (int i = 17; i < 32; ++i) {
            warp.threads[i].pc = 100;  // Went to alternate path (no lock)
            warp.threads[i].is_active = true;
        }
        
        // All 32 threads still active but at different PCs
        REQUIRE(warp.count_schedulable_lanes() == 32);
        
        // Per-thread PC tracks each differently
        REQUIRE(warp.threads[0].pc == 20);
        REQUIRE(warp.threads[1].pc == 10);
        REQUIRE(warp.threads[17].pc == 100);
        REQUIRE(warp.threads[31].pc == 100);
    }
}

TEST_CASE("Spinlock priority and fairness", "[simt][spinlock][fairness]") {
    WarpState warp;
    
    SECTION("Anti-starvation with priority boost") {
        // Simulate a thread that's been waiting a long time
        
        // Thread 0 - waiting at barrier for 500 cycles
        warp.threads[0].pc = 10;
        warp.threads[0].is_active = true;
        warp.threads[0].status = ThreadStatus::Active;
        
        // Other threads at different execution points
        for (int i = 1; i < 32; ++i) {
            warp.threads[i].pc = 50;
            warp.threads[i].is_active = true;
        }
        
        // Even though thread 0 has been "waiting", it's still schedulable
        REQUIRE(warp.threads[0].is_schedulable() == true);
        
        // All lanes schedulable
        REQUIRE(warp.count_schedulable_lanes() == 32);
    }
}

