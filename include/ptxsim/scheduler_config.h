#ifndef SCHEDULER_CONFIG_H
#define SCHEDULER_CONFIG_H

#include <cstdint>
#include <string>
#include <sstream>
#include <fstream>
#include <map>

/**
 * @file scheduler_config.h
 * @brief Configuration parser for priority-based anti-starvation scheduler
 * @details Reads scheduler_config.ini and provides type-safe access to parameters
 * @author PTX-EMU Team
 * @date 2026-04-02
 */

namespace ptxsim {

// Scheduler algorithm enum
enum class SchedulerAlgorithm {
    RoundRobin,
    Priority,
    GTO,  // Greedy Then Oldest
    LRR   // Least Recently Run
};

// Convert string to algorithm
inline SchedulerAlgorithm parse_algorithm(const std::string& str) {
    if (str == "round_robin") return SchedulerAlgorithm::RoundRobin;
    if (str == "priority") return SchedulerAlgorithm::Priority;
    if (str == "gto") return SchedulerAlgorithm::GTO;
    if (str == "lrr") return SchedulerAlgorithm::LRR;
    return SchedulerAlgorithm::Priority;  // Default
}

// Scheduler configuration structure
struct SchedulerConfig {
    // Priority weights
    uint32_t priority_memory_wait = 100;
    uint32_t priority_no_dependency = 50;
    uint32_t priority_barrier_waiting = 25;
    uint32_t priority_wait_cycle_scale = 1;
    uint32_t priority_max_wait_bonus = 200;
    
    // Algorithm selection
    SchedulerAlgorithm algorithm = SchedulerAlgorithm::Priority;
    
    // Anti-starvation
    bool anti_starvation_enabled = true;
    
    // Time slicing
    uint32_t time_slice_cycles = 0;
    
    // Warp management
    uint32_t min_active_warps = 2;
    
    // Barrier-aware scheduling
    bool barrier_aware_enabled = true;
    uint32_t barrier_full_barrier_boost = 50;
    uint32_t barrier_check_interval = 1;
    
    // Memory settings (for scheduling decisions)
    uint32_t max_outstanding_requests = 8;
    uint32_t global_mem_latency = 400;
    uint32_t shared_mem_latency = 1;
    
    // Debug/logging
    bool verbose = false;
    bool log_scheduling = false;
    bool log_priorities = false;
    uint32_t report_interval = 1000;
    
    /**
     * Load configuration from INI file
     * @param filename Path to scheduler_config.ini
     * @return true if loaded successfully, false on error
     */
    bool load_from_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;  // Use defaults
        }
        
        std::string line;
        std::string current_section;
        
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#' || line[0] == ';') {
                continue;
            }
            
            // Section header
            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.length() - 2);
                continue;
            }
            
            // Key=value pair
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) {
                continue;
            }
            
            std::string key = trim(line.substr(0, eq_pos));
            std::string value = trim(line.substr(eq_pos + 1));
            
            parse_value(current_section, key, value);
        }
        
        file.close();
        return true;
    }
    
    /**
     * Get warp priority based on state
     * @param is_memory_wait True if warp is waiting on memory
     * @param is_barrier_wait True if warp is waiting at barrier
     * @param wait_cycles Number of cycles warp has been waiting
     * @return Computed priority value
     */
    uint32_t compute_warp_priority(
        bool is_memory_wait,
        bool is_barrier_wait,
        uint32_t wait_cycles) const 
    {
        uint32_t priority = is_memory_wait ? priority_memory_wait : 
                           (is_barrier_wait ? priority_barrier_waiting : 
                                              priority_no_dependency);
        
        // Anti-starvation: boost priority based on wait time
        if (anti_starvation_enabled && wait_cycles > 0) {
            uint32_t wait_bonus = wait_cycles * priority_wait_cycle_scale;
            if (wait_bonus > priority_max_wait_bonus) {
                wait_bonus = priority_max_wait_bonus;
            }
            priority += wait_bonus;
        }
        
        return priority;
    }
    
    /**
     * Convert config to string for debugging
     * @return Human-readable configuration summary
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << "Scheduler Configuration:\n";
        oss << "  Algorithm: " << algorithm_to_string(algorithm) << "\n";
        oss << "  Priority.memory_wait: " << priority_memory_wait << "\n";
        oss << "  Priority.no_dependency: " << priority_no_dependency << "\n";
        oss << "  Priority.barrier_waiting: " << priority_barrier_waiting << "\n";
        oss << "  Anti-starvation: " << (anti_starvation_enabled ? "enabled" : "disabled") << "\n";
        oss << "  Wait cycle scale: " << priority_wait_cycle_scale << "\n";
        oss << "  Max wait bonus: " << priority_max_wait_bonus << "\n";
        oss << "  Barrier-aware: " << (barrier_aware_enabled ? "enabled" : "disabled") << "\n";
        return oss.str();
    }
    
private:
    // Trim whitespace from string
    static std::string trim(const std::string& str) {
        size_t start = str.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        size_t end = str.find_last_not_of(" \t\n\r");
        return str.substr(start, end - start + 1);
    }
    
    // Parse value based on section and key
    void parse_value(const std::string& section, const std::string& key, const std::string& value) {
        if (section == "scheduler") {
            if (key == "priority.memory_wait") priority_memory_wait = std::stoul(value);
            else if (key == "priority.no_dependency") priority_no_dependency = std::stoul(value);
            else if (key == "priority.barrier_waiting") priority_barrier_waiting = std::stoul(value);
            else if (key == "priority.wait_cycle_scale") priority_wait_cycle_scale = std::stoul(value);
            else if (key == "priority.max_wait_bonus") priority_max_wait_bonus = std::stoul(value);
            else if (key == "algorithm") algorithm = parse_algorithm(value);
            else if (key == "anti_starvation_enabled") anti_starvation_enabled = (value == "true");
            else if (key == "time_slice_cycles") time_slice_cycles = std::stoul(value);
            else if (key == "min_active_warps") min_active_warps = std::stoul(value);
        }
        else if (section == "barrier_aware") {
            if (key == "enabled") barrier_aware_enabled = (value == "true");
            else if (key == "full_barrier_boost") barrier_full_barrier_boost = std::stoul(value);
            else if (key == "check_interval_cycles") barrier_check_interval = std::stoul(value);
        }
        else if (section == "memory") {
            if (key == "max_outstanding_requests") max_outstanding_requests = std::stoul(value);
            else if (key == "global_mem_latency") global_mem_latency = std::stoul(value);
            else if (key == "shared_mem_latency") shared_mem_latency = std::stoul(value);
        }
        else if (section == "debug") {
            if (key == "verbose") verbose = (value == "true");
            else if (key == "log_scheduling") log_scheduling = (value == "true");
            else if (key == "log_priorities") log_priorities = (value == "true");
            else if (key == "report_interval") report_interval = std::stoul(value);
        }
    }
    
    // Convert algorithm enum to string
    static std::string algorithm_to_string(SchedulerAlgorithm algo) {
        switch (algo) {
            case SchedulerAlgorithm::RoundRobin: return "round_robin";
            case SchedulerAlgorithm::Priority: return "priority";
            case SchedulerAlgorithm::GTO: return "gto";
            case SchedulerAlgorithm::LRR: return "lrr";
            default: return "unknown";
        }
    }
};

// Singleton instance for global access
class SchedulerConfigManager {
public:
    static SchedulerConfigManager& instance() {
        static SchedulerConfigManager inst;
        return inst;
    }
    
    const SchedulerConfig& get_config() const { return config_; }
    SchedulerConfig& get_config() { return config_; }
    
    bool load_config(const std::string& filename) {
        return config_.load_from_file(filename);
    }
    
private:
    SchedulerConfigManager() {
        // Try to load from default location
        config_.load_from_file("configs/scheduler_config.ini");
    }
    SchedulerConfig config_;
};

// Convenience function
inline const SchedulerConfig& get_scheduler_config() {
    return SchedulerConfigManager::instance().get_config();
}

} // namespace ptxsim

#endif // SCHEDULER_CONFIG_H
