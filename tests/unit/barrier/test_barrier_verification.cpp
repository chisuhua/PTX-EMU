#include "catch_amalgamated.hpp"
#include "ptxsim/wbar.h"
#include "ptxsim/thread_state.h"
#include "ptxsim/simt_stack.h"
#include <array>

using namespace ptxsim;

TEST_CASE("Memory Fence Verification", "[barrier][fence][phase4]") {
    Wbar wbar;
    
    SECTION("Wbar initialization") {
        wbar.init(0xFFFFFFFF, 100);
        REQUIRE(!wbar.is_complete());
    }
    
    SECTION("All lanes arrive") {
        wbar.init(0xFFFFFFFF, 100);
        
        for (int i = 0; i < 32; i++) {
            wbar.arrive(i);
        }
        
        REQUIRE(wbar.is_complete());
    }
    
    SECTION("Partial arrive not complete") {
        wbar.init(0xFFFFFFFF, 100);
        
        for (int i = 0; i < 16; i++) {
            wbar.arrive(i);
        }
        
        REQUIRE(!wbar.is_complete());
    }
    
    SECTION("Dynamic participation mask") {
        wbar.init(0xFFFFFFFF, 100);
        
        wbar.arrive(0);
        REQUIRE((wbar.participation_mask & 0x1) != 0);
        
        wbar.arrive(1);
        REQUIRE((wbar.participation_mask & 0x3) == 0x3);
    }
}

TEST_CASE("Barrier Memory Fence Verification", "[barrier][verification][phase4]") {
    Wbar wbar;
    
    SECTION("Debug mode verification enabled") {
        #ifdef PTX_DEBUG
        wbar.enable_memory_fence_verification(true);
        REQUIRE(wbar.is_memory_fence_verification_enabled());
        #endif
    }
    
    SECTION("Pre-barrier store tracking") {
        #ifdef PTX_DEBUG
        wbar.enable_memory_fence_verification(true);
        
        wbar.init(100, 0xFFFFFFFF);
        wbar.record_pre_barrier_store(0, 0x1000);
        wbar.record_pre_barrier_store(1, 0x1004);
        
        REQUIRE(wbar.get_pre_barrier_store_count() == 2);
        #endif
    }
    
    SECTION("Store visibility check") {
        #ifdef PTX_DEBUG
        wbar.enable_memory_fence_verification(true);
        wbar.init(100, 0xFFFFFFFF);
        
        wbar.record_pre_barrier_store(0, 0x1000);
        
        for (int i = 0; i < 32; i++) {
            wbar.arrive(i);
        }
        
        bool all_visible = wbar.verify_all_stores_visible();
        REQUIRE(all_visible);
        #endif
    }
}

TEST_CASE("SIMT Stack Barrier Integration", "[simt][barrier][phase4]") {
    SIMTStack simt_stack;
    
    SECTION("Branch pushes to stack") {
        SIMTStackEntry entry;
        entry.branch_pc = 10;
        entry.reconvergence_pc = 100;
        entry.active_mask = 0xFFFF;
        entry.return_mask = 0xFFFFFFFF;
        entry.return_pc = 100;
        
        simt_stack.push(entry);
        
        REQUIRE(!simt_stack.empty());
        REQUIRE(simt_stack.depth() == 1);
        REQUIRE(simt_stack.top().branch_pc == 10);
    }
    
    SECTION("Reconvergence before barrier") {
        std::array<ThreadState, 32> threads;
        threads.fill(ThreadState());
        
        for (int i = 0; i < 32; i++) {
            threads[i].pc = 100;
        }
        
        SIMTStackEntry entry;
        entry.branch_pc = 10;
        entry.reconvergence_pc = 100;
        entry.return_mask = 0xFFFFFFFF;
        
        simt_stack.push(entry);
        
        bool converged = simt_stack.check_reconvergence(threads);
        
        REQUIRE(converged);
        REQUIRE(simt_stack.empty());
    }
}

TEST_CASE("Barrier Semantic Verification", "[barrier][semantics][phase4]") {
    Wbar wbar;
    
    SECTION("Barrier complete after all arrive") {
        wbar.init(0xFFFFFFFF, 50);
        
        for (int i = 0; i < 32; i++) {
            wbar.arrive(i);
        }
        
        REQUIRE(wbar.is_complete());
        REQUIRE(wbar.reconvergence_pc == 50);
    }
    
    SECTION("Barrier reset after complete") {
        wbar.init(0xFFFFFFFF, 50);
        
        for (int i = 0; i < 32; i++) {
            wbar.arrive(i);
        }
        
        wbar.reset();
        
        REQUIRE(!wbar.is_complete());
        REQUIRE(wbar.participation_mask == 0);
        REQUIRE(wbar.arrived_mask == 0);
    }
}
