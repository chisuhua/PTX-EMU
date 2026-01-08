#include "memory/resource_manager.h"
#include "memory/shared_memory_manager.h"
#include "utils/logger.h"

ResourceManager& ResourceManager::instance() {
    static ResourceManager instance;
    return instance;
}

void ResourceManager::initialize(int sm_count, size_t max_shared_mem_per_sm) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (shared_mem_managers_.size() > 0) {
        PTX_DEBUG_EMU("ResourceManager already initialized with %zu SMs, skipping re-initialization", 
                      shared_mem_managers_.size());
        return;
    }
    
    shared_mem_managers_.reserve(sm_count);
    for (int i = 0; i < sm_count; ++i) {
        shared_mem_managers_.push_back(
            std::make_unique<SharedMemoryManager>(max_shared_mem_per_sm));
    }
    
    PTX_DEBUG_EMU("Initialized ResourceManager with %d SMs, each with %zu bytes shared memory", 
                  sm_count, max_shared_mem_per_sm);
}

SharedMemoryManager* ResourceManager::get_shared_memory_manager(int sm_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (sm_id < 0 || sm_id >= static_cast<int>(shared_mem_managers_.size())) {
        PTX_DEBUG_EMU("Invalid SM ID: %d, available SMs: %zu", 
                      sm_id, shared_mem_managers_.size());
        return nullptr;
    }
    
    return shared_mem_managers_[sm_id].get();
}

void ResourceManager::print_resource_usage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    PTX_DEBUG_EMU("=== Resource Manager Statistics ===");
    for (size_t i = 0; i < shared_mem_managers_.size(); ++i) {
        auto* mgr = shared_mem_managers_[i].get();
        if (mgr) {
            size_t allocated = mgr->get_allocated_size();
            size_t max_size = mgr->get_max_size();
            size_t available = mgr->get_available_size();
            
            PTX_DEBUG_EMU("SM %zu: Allocated=%zu, Max=%zu, Available=%zu, Usage=%.2f%%", 
                          i, allocated, max_size, available, 
                          (max_size > 0) ? (100.0 * allocated / max_size) : 0.0);
        }
    }
    PTX_DEBUG_EMU("===================================");
}