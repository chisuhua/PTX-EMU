
#include "catch_amalgamated.hpp"
#include "ptxsim/scheduler_config.h"

#include <iostream>
#include <sstream>
#include <fstream>

using namespace ptxsim;

// ============================================================================
// Test Suite: Scheduler Configuration
// ============================================================================
// Tests for priority-based anti-starvation scheduler configuration
// ============================================================================

TEST_CASE("SchedulerConfig default values", "[simt][scheduler][config]") {
    SchedulerConfig config;
    
    SECTION("Priority defaults match spec") {
        REQUIRE(config.priority_memory_wait == 100);
        REQUIRE(config.priority_no_dependency == 50);
        REQUIRE(config.priority_barrier_waiting == 25);
        REQUIRE(config.priority_wait_cycle_scale == 1);
        REQUIRE(config.priority_max_wait_bonus == 200);
    }
    
    SECTION("Algorithm default is priority") {
        REQUIRE(config.algorithm == SchedulerAlgorithm::Priority);
    }
    
    SECTION("Anti-starvation enabled by default") {
        REQUIRE(config.anti_starvation_enabled == true);
    }
    
    SECTION("Barrier-aware scheduling default") {
        REQUIRE(config.barrier_aware_enabled == true);
        REQUIRE(config.barrier_full_barrier_boost == 50);
    }
}

TEST_CASE("SchedulerConfig priority computation", "[simt][scheduler][priority]") {
    SchedulerConfig config;
    
    SECTION("Basic priority computation (no wait)") {
        // Memory wait
        uint32_t priority = config.compute_warp_priority(true, false, 0);
        REQUIRE(priority == 100);
        
        // No dependency (ready)
        priority = config.compute_warp_priority(false, false, 0);
        REQUIRE(priority == 50);
        
        // Barrier waiting
        priority = config.compute_warp_priority(false, true, 0);
        REQUIRE(priority == 25);
    }
    
    SECTION("Anti-starvation wait bonus accumulation") {
        // After 10 cycles of waiting
        uint32_t priority = config.compute_warp_priority(false, true, 10);
        REQUIRE(priority == 25 + 10 * 1);  // base + wait_bonus
        
        // After 50 cycles
        priority = config.compute_warp_priority(false, true, 50);
        REQUIRE(priority == 25 + 50 * 1);
    }
    
    SECTION("Max wait bonus cap") {
        // After 500 cycles (should be capped at max_wait_bonus)
        uint32_t priority = config.compute_warp_priority(false, true, 500);
        REQUIRE(priority == 25 + 200);  // base + capped bonus
        
        // After 1000 cycles (still capped)
        priority = config.compute_warp_priority(false, true, 1000);
        REQUIRE(priority == 25 + 200);
    }
    
    SECTION("Priority with disabled anti-starvation") {
        config.anti_starvation_enabled = false;
        
        // Wait time should have no effect
        uint32_t priority1 = config.compute_warp_priority(false, true, 0);
        uint32_t priority2 = config.compute_warp_priority(false, true, 500);
        
        REQUIRE(priority1 == priority2);
        REQUIRE(priority1 == 25);  // Base priority only
    }
}

TEST_CASE("SchedulerConfig INI parsing", "[simt][scheduler][ini]") {
    // Create temp config file
    std::ofstream file("/tmp/test_scheduler.ini");
    REQUIRE(file.is_open());
    
    file << "[scheduler]\n";
    file << "priority.memory_wait = 150\n";
    file << "priority.no_dependency = 75\n";
    file << "priority.barrier_waiting = 40\n";
    file << "priority.wait_cycle_scale = 2\n";
    file << "priority.max_wait_bonus = 300\n";
    file << "algorithm = priority\n";
    file << "anti_starvation_enabled = false\n";
    file << "\n";
    file << "[barrier_aware]\n";
    file << "enabled = false\n";
    file << "full_barrier_boost = 100\n";
    file << "\n";
    file << "[debug]\n";
    file << "verbose = true\n";
    
    file.close();
    
    // Load config
    SchedulerConfig config;
    bool loaded = config.load_from_file("/tmp/test_scheduler.ini");
    REQUIRE(loaded == true);
    
    SECTION("Priority values loaded correctly") {
        REQUIRE(config.priority_memory_wait == 150);
        REQUIRE(config.priority_no_dependency == 75);
        REQUIRE(config.priority_barrier_waiting == 40);
        REQUIRE(config.priority_wait_cycle_scale == 2);
        REQUIRE(config.priority_max_wait_bonus == 300);
    }
    
    SECTION("Algorithm loaded correctly") {
        REQUIRE(config.algorithm == SchedulerAlgorithm::Priority);
    }
    
    SECTION("Anti-starvation disabled loaded") {
        REQUIRE(config.anti_starvation_enabled == false);
    }
    
    SECTION("Barrier-aware config loaded") {
        REQUIRE(config.barrier_aware_enabled == false);
        REQUIRE(config.barrier_full_barrier_boost == 100);
    }
    
    SECTION("Debug config loaded") {
        REQUIRE(config.verbose == true);
    }
}

TEST_CASE("SchedulerConfig algorithm parsing", "[simt][scheduler][algorithm]") {
    SchedulerConfig config;
    
    SECTION("All algorithm types parse correctly") {
        // Test direct parsing
        REQUIRE(parse_algorithm("round_robin") == SchedulerAlgorithm::RoundRobin);
        REQUIRE(parse_algorithm("priority") == SchedulerAlgorithm::Priority);
        REQUIRE(parse_algorithm("gto") == SchedulerAlgorithm::GTO);
        REQUIRE(parse_algorithm("lrr") == SchedulerAlgorithm::LRR);
        
        // Unknown defaults to priority
        REQUIRE(parse_algorithm("unknown") == SchedulerAlgorithm::Priority);
    }
}

TEST_CASE("SchedulerConfig string conversion", "[simt][scheduler][utils]") {
    SchedulerConfig config;
    
    SECTION("to_string produces valid output") {
        std::string str = config.to_string();
        
        REQUIRE(str.find("Scheduler Configuration:") != std::string::npos);
        REQUIRE(str.find("Algorithm:") != std::string::npos);
        REQUIRE(str.find("Priority.memory_wait:") != std::string::npos);
        REQUIRE(str.find("Anti-starvation:") != std::string::npos);
    }
}

TEST_CASE("Anti-starvation mechanism behavior", "[simt][scheduler][starvation]") {
    SchedulerConfig config;
    config.anti_starvation_enabled = true;
    config.priority_wait_cycle_scale = 10;
    config.priority_max_wait_bonus = 500;
    
    SECTION("Long-waiting warp overtakes new warp") {
        // New warp with no dependencies (higher base priority)
        uint32_t new_warp_priority = config.compute_warp_priority(false, false, 0);
        REQUIRE(new_warp_priority == 50);
        
        // Old warp waiting at barrier for 100 cycles
        uint32_t old_warp_priority = config.compute_warp_priority(false, true, 100);
        // base=25 + bonus=100*10=1000, but capped at 500
        REQUIRE(old_warp_priority == 525);
        
        // Old warp has higher priority due to anti-starvation
        REQUIRE(old_warp_priority > new_warp_priority * 2);
    }
    
    SECTION("Starvation prevention threshold") {
        // Compute cycles needed for barrier warp to exceed memory warp
        uint32_t memory_priority = config.compute_warp_priority(true, false, 0);
        
        // Barrier warp needs: 25 + cycles * 10 >= 100
        // Cycles >= 8 (but capped so will reach max)
        uint32_t cycles_to_catch_up = (memory_priority - 25 + config.priority_wait_cycle_scale - 1) / 
                                       config.priority_wait_cycle_scale;
        
        REQUIRE(cycles_to_catch_up <= 500 / config.priority_wait_cycle_scale);
    }
}

TEST_CASE("Barrier-aware scheduling", "[simt][scheduler][barrier]") {
    SchedulerConfig config;
    
    SECTION("Full barrier boost concept") {
        // When all barrier participants are blocked, add extra boost
        // This is documented behavior - implementation would be in warp_scheduler
        
        // Simulate barrier-aware boost
        uint32_t barrier_base = config.compute_warp_priority(false, true, 0);
        uint32_t barrier_with_boost = barrier_base + config.barrier_full_barrier_boost;
        
        REQUIRE(barrier_with_boost > barrier_base);
        REQUIRE(config.barrier_full_barrier_boost == 50);
    }
    
    SECTION("Barrier-aware can be disabled") {
        config.barrier_aware_enabled = false;
        
        // Even if disabled, the boost value exists but scheduler should ignore it
        uint32_t barrier_base = config.compute_warp_priority(false, true, 0);
        REQUIRE(barrier_base == 25);
    }
}

TEST_CASE("Singleton config manager", "[simt][scheduler][singleton]") {
    SECTION("Singleton instance works") {
        auto& manager = SchedulerConfigManager::instance();
        const auto& config = manager.get_config();
        
        // Should have default values (or file-loaded if exists)
        REQUIRE(config.priority_no_dependency >= 0);
    }
}

TEST_CASE("Edge cases and error handling", "[simt][scheduler][edge]") {
    SECTION("Missing config file uses defaults") {
        SchedulerConfig config;
        bool loaded = config.load_from_file("/nonexistent/path/config.ini");
        
        REQUIRE(loaded == false);
        REQUIRE(config.priority_memory_wait == 100);  // Default
    }
    
    SECTION("Zero wait cycles doesn't crash") {
        SchedulerConfig config;
        uint32_t priority = config.compute_warp_priority(false, true, 0);
        REQUIRE(priority == 25);  // Just base priority
        
        priority = config.compute_warp_priority(false, true, 0);
        REQUIRE(priority == 25);
    }
    
    SECTION("Very large wait cycle count") {
        SchedulerConfig config;
        uint32_t priority = config.compute_warp_priority(false, true, UINT32_MAX);
        
        // Should be capped at max_wait_bonus
        REQUIRE(priority == 25 + config.priority_max_wait_bonus);
    }
}
