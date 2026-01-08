#ifndef SHARED_MEMORY_MANAGER_H
#define SHARED_MEMORY_MANAGER_H

#include <vector>
#include <mutex>
#include <map>

class SharedMemoryManager {
public:
    explicit SharedMemoryManager(size_t max_shared_mem);
    ~SharedMemoryManager();
    
    // 分配共享内存
    void* allocate(size_t size, int block_id);
    
    // 释放共享内存
    void deallocate(void* ptr, int block_id);
    
    // 获取已分配内存大小
    size_t get_allocated_size() const;
    
    // 获取剩余可用内存
    size_t get_available_size() const;
    
    // 获取最大内存大小
    size_t get_max_size() const { return max_shared_mem_; }
    
private:
    struct BlockInfo {
        void* ptr;
        size_t size;
        int block_id;
    };
    
    size_t max_shared_mem_;
    size_t allocated_size_;
    std::vector<BlockInfo> allocated_blocks_;
    mutable std::mutex mutex_;
};

#endif // SHARED_MEMORY_MANAGER_H