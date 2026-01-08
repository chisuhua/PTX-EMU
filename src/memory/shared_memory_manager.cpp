#include "memory/shared_memory_manager.h"
#include "utils/logger.h"
#include <cstring>

SharedMemoryManager::SharedMemoryManager(size_t max_shared_mem) 
    : max_shared_mem_(max_shared_mem), allocated_size_(0) {
}

SharedMemoryManager::~SharedMemoryManager() {
    // 检查是否还有未释放的内存
    if (allocated_size_ > 0) {
        PTX_DEBUG_EMU("Warning: SharedMemoryManager destroyed with %zu bytes "
                      "of allocated shared memory still remaining", 
                      allocated_size_);
    }
    
    // 清理所有已分配的内存
    for (auto& block : allocated_blocks_) {
        if (block.ptr) {
            free(block.ptr);
        }
    }
    allocated_blocks_.clear();
}

void* SharedMemoryManager::allocate(size_t size, int block_id) {
    if (size == 0) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 检查是否有足够的空间
    if (allocated_size_ + size > max_shared_mem_) {
        PTX_DEBUG_EMU("Insufficient shared memory: requested %zu, available %zu", 
                      size, get_available_size());
        return nullptr;
    }
    
    // 分配内存
    void* ptr = malloc(size);
    if (!ptr) {
        PTX_DEBUG_EMU("Failed to allocate shared memory of size %zu", size);
        return nullptr;
    }
    
    // 记录分配信息
    allocated_blocks_.push_back({ptr, size, block_id});
    allocated_size_ += size;
    
    PTX_DEBUG_EMU("Allocated %zu bytes of shared memory at %p for block %d, "
                  "total allocated: %zu", size, ptr, block_id, allocated_size_);
    
    return ptr;
}

void SharedMemoryManager::deallocate(void* ptr, int block_id) {
    if (!ptr) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 查找对应的分配记录
    auto it = allocated_blocks_.begin();
    while (it != allocated_blocks_.end()) {
        if (it->ptr == ptr && it->block_id == block_id) {
            allocated_size_ -= it->size;
            
            free(it->ptr);
            allocated_blocks_.erase(it);
            
            PTX_DEBUG_EMU("Deallocated %zu bytes of shared memory at %p for block %d, "
                          "total allocated: %zu", it->size, ptr, block_id, allocated_size_);
            return;
        }
        ++it;
    }
    
    PTX_DEBUG_EMU("Warning: Attempted to deallocate unknown shared memory pointer %p", ptr);
}

size_t SharedMemoryManager::get_allocated_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_size_;
}

size_t SharedMemoryManager::get_available_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_shared_mem_ - allocated_size_;
}