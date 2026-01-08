#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include <vector>
#include <memory>
#include <mutex>

class SharedMemoryManager;

class ResourceManager {
public:
    static ResourceManager& instance();
    
    // 初始化资源管理器，指定SM数量
    void initialize(int sm_count, size_t max_shared_mem_per_sm);
    
    // 获取共享内存管理器
    SharedMemoryManager* get_shared_memory_manager(int sm_id);
    
    // 资源统计
    void print_resource_usage() const;
    
private:
    ResourceManager() = default;
    ~ResourceManager() = default;
    
    std::vector<std::unique_ptr<SharedMemoryManager>> shared_mem_managers_;
    mutable std::mutex mutex_;
};

#endif // RESOURCE_MANAGER_H