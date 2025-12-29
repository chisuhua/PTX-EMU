#include "memory/simple_memory_allocator.h"
#include "utils/logger.h"

SimpleMemoryAllocator::SimpleMemoryAllocator() : pool_size_(GLOBAL_SIZE), last_allocated_end_(0) {
    // 分配内存池（使用mmap虚拟内存）
    pool_ = static_cast<uint8_t *>(
        mmap(nullptr, pool_size_, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (pool_ == MAP_FAILED) {
        throw std::runtime_error("Failed to mmap memory pool");
    }
    
    // 初始化一个大的空闲块
    free_blocks_.insert({pool_size_, 0});
}

SimpleMemoryAllocator::~SimpleMemoryAllocator() {
    if (pool_ && pool_ != MAP_FAILED) {
        munmap(pool_, pool_size_);
    }
}

size_t SimpleMemoryAllocator::allocate(size_t size) {
    if (size == 0) {
        return static_cast<size_t>(-1); // 返回无效偏移
    }

    std::lock_guard<std::mutex> lock(mutex_);
    
    // 查找第一个足够大的空闲块（first-fit）
    auto free_it = free_blocks_.lower_bound(size);
    if (free_it == free_blocks_.end()) {
        PTX_DEBUG_MEM("Memory allocation failed: requested size %zu, no suitable free block found", size);
        return static_cast<size_t>(-1);
    }
    
    size_t free_size = free_it->first;
    size_t free_offset = free_it->second;
    
    // 从空闲列表中移除这个块
    free_blocks_.erase(free_it);
    
    // 检查是否需要分割块
    if (free_size > size) {
        // 将剩余部分作为新的空闲块插入
        size_t remaining_offset = free_offset + size;
        size_t remaining_size = free_size - size;
        free_blocks_.insert({remaining_size, remaining_offset});
    }
    
    // 记录已分配的块
    allocated_blocks_[free_offset] = {free_offset, size};
    last_allocated_end_ = free_offset + size;
    
    PTX_DEBUG_MEM("Memory allocated: offset=%zu, size=%zu", free_offset, size);
    return free_offset;
}

void SimpleMemoryAllocator::deallocate(size_t offset) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocated_blocks_.find(offset);
    if (it != allocated_blocks_.end()) {
        Allocation alloc = it->second;
        
        // 从已分配列表中移除
        allocated_blocks_.erase(it);
        
        // 尝试与相邻的空闲块合并
        size_t new_offset = alloc.offset;
        size_t new_size = alloc.size;
        
        // 检查低地址是否有相邻的空闲块
        bool merged_with_lower = false;
        for (auto free_it = free_blocks_.begin(); free_it != free_blocks_.end(); ++free_it) {
            size_t free_offset = free_it->second;
            size_t free_size = free_it->first;
            
            if (free_offset + free_size == new_offset) {
                // 与低地址空闲块合并
                new_offset = free_offset;
                new_size += free_size;
                
                free_blocks_.erase(free_it);
                merged_with_lower = true;
                break;
            }
        }
        
        // 检查高地址是否有相邻的空闲块
        bool merged_with_higher = false;
        for (auto free_it = free_blocks_.begin(); free_it != free_blocks_.end(); ++free_it) {
            size_t free_offset = free_it->second;
            
            if (new_offset + new_size == free_offset) {
                // 与高地址空闲块合并
                new_size += free_it->first;
                
                free_blocks_.erase(free_it);
                merged_with_higher = true;
                break;
            }
        }
        
        // 插入合并后的空闲块
        free_blocks_.insert({new_size, new_offset});
        
        PTX_DEBUG_MEM("Memory freed: offset=%zu, size=%zu", offset, alloc.size);
    } else {
        PTX_DEBUG_MEM("Attempt to free unallocated memory: offset=%zu", offset);
    }
}

SimpleMemoryAllocator::Allocation* SimpleMemoryAllocator::get_allocation(size_t offset) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocated_blocks_.find(offset);
    if (it != allocated_blocks_.end()) {
        return &it->second;
    }
    return nullptr;
}