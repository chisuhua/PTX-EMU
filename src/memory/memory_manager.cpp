// memory/memory_manager.cpp
#include "memory/memory_manager.h"
#include "memory/simple_memory_allocator.h"
#include "memory/memory_interface.h" // 添加缺少的头文件
#include "utils/logger.h" // 添加日志头文件
#include <cstring>
#include <stdexcept>
#include <unistd.h>
#include <mutex> // 添加mutex头文件
#include <unordered_map> // 添加unordered_map头文件

namespace {
    SimpleMemoryAllocator* get_global_allocator() {
        static SimpleMemoryAllocator allocator;
        return &allocator;
    }
}

MemoryManager &MemoryManager::instance() {
    static MemoryManager inst;
    return inst;
}

MemoryManager::MemoryManager() {
    // 使用全局内存分配器
    allocator_ = get_global_allocator();
    
    // 初始化共享内存池
    shared_pool_ = new uint8_t[SHARED_SIZE];
}

MemoryManager::~MemoryManager() {
    // 释放共享内存池
    if (shared_pool_) {
        delete[] shared_pool_;
        shared_pool_ = nullptr;
    }
    // 不需要释放allocator_，因为它是一个全局单例
}

void MemoryManager::set_memory_interface(MemoryInterface *mem_if) {
    std::lock_guard<std::mutex> lock(mutex_);
    memory_interface_ = mem_if;
}

void *MemoryManager::malloc(size_t size) { return malloc_managed(size); }

void *MemoryManager::malloc_managed(size_t size) {
    if (size == 0)
        return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t offset = allocator_->allocate(size);
    if (offset == static_cast<size_t>(-1)) {
        return nullptr;
    }

    void *host_ptr = allocator_->get_pool() + offset;
    allocations_[reinterpret_cast<uint64_t>(host_ptr)] = {offset, size};
    
    PTX_DEBUG_MEM("GLOBAL memory allocated: ptr=%p, offset=%zu, size=%zu", 
        host_ptr, offset, size);
    return host_ptr;
}

mycudaError_t MemoryManager::free(void *dev_ptr) {
    if (!dev_ptr)
        return Success;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(reinterpret_cast<uint64_t>(dev_ptr));
    if (it != allocations_.end()) {
        allocator_->deallocate(it->second.offset);
        allocations_.erase(it);
        return Success;
    }
    return ErrorInvalidValue;
}

void *MemoryManager::malloc_param(size_t size) {
    if (size == 0)
        return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t offset = allocator_->allocate(size);
    if (offset == static_cast<size_t>(-1)) {
        return nullptr;
    }

    void *param_ptr = allocator_->get_pool() + offset;
    param_allocations_[reinterpret_cast<uint64_t>(param_ptr)] = {offset, size};
    
    PTX_DEBUG_MEM("PARAM memory allocated: ptr=%p, offset=%zu, size=%zu", 
        param_ptr, offset, size);
    return param_ptr;
}

void MemoryManager::free_param(void *param_ptr) {
    if (!param_ptr)
        return;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = param_allocations_.find(reinterpret_cast<uint64_t>(param_ptr));
    if (it != param_allocations_.end()) {
        allocator_->deallocate(it->second.offset);
        param_allocations_.erase(it);
    }
}

void *MemoryManager::malloc_shared(size_t size) {
    if (size == 0)
        return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);
    
    // 在共享内存池中查找可用空间
    size_t current_offset = 0;
    bool found = false;
    
    // 简单的连续分配策略，实际应用中可能需要更复杂的内存管理
    for (auto& pair : shared_allocations_) {
        size_t end_offset = pair.second.offset + pair.second.size;
        if (end_offset > current_offset) {
            current_offset = end_offset;
        }
    }
    
    // 检查是否有足够的空间
    if (current_offset + size > SHARED_SIZE) {
        PTX_DEBUG_MEM("Not enough shared memory: requested=%zu, available=%zu", 
            size, SHARED_SIZE - current_offset);
        return nullptr;
    }
    
    void *shared_ptr = shared_pool_ + current_offset;
    shared_allocations_[reinterpret_cast<uint64_t>(shared_ptr)] = {current_offset, size};
    
    PTX_DEBUG_MEM("SHARED memory allocated: ptr=%p, offset=%zu, size=%zu", 
        shared_ptr, current_offset, size);
    return shared_ptr;
}

void MemoryManager::free_shared(void *shared_ptr) {
    if (!shared_ptr)
        return;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = shared_allocations_.find(reinterpret_cast<uint64_t>(shared_ptr));
    if (it != shared_allocations_.end()) {
        // 对于简单实现，我们不真正释放空间，只是移除记录
        // 实际应用中可能需要实现真正的内存块回收
        shared_allocations_.erase(it);
    }
}

void MemoryManager::access(void *host_ptr, void *data, size_t size,
                           bool is_write) {
    access(host_ptr, data, size, is_write, MemorySpace::GLOBAL);
}

void MemoryManager::access(void *host_ptr, void *data, size_t size,
                           bool is_write, MemorySpace space) {
    if (!host_ptr || !data || size == 0) {
        throw std::invalid_argument("Invalid memory access arguments");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!memory_interface_) {
        throw std::runtime_error("MemoryInterface not initialized");
    }

    // 根据地址空间类型处理访问
    switch (space) {
    case MemorySpace::GLOBAL: {
        // 检查host_ptr是否在任何已分配的区间内
        bool found = false;
        MemoryManager::Allocation* alloc = nullptr;
        
        for (auto& pair : allocations_) {
            uint8_t* start_addr = allocator_->get_pool() + pair.second.offset;
            uint8_t* end_addr = start_addr + pair.second.size;
            uint8_t* access_addr = static_cast<uint8_t*>(host_ptr);
            
            if (access_addr >= start_addr && access_addr < end_addr) {
                // 确保访问的范围没有超出分配的范围
                if (access_addr + size <= end_addr) {
                    alloc = &pair.second;
                    found = true;
                    break;
                } else {
                    PTX_DEBUG_MEM("Buffer overflow in GLOBAL memory access: ptr=%p, size=%zu, access range=[%p, %p], allocated range=[%p, %p]", 
                        host_ptr, size, access_addr, access_addr + size, start_addr, end_addr);
                    throw std::runtime_error("Buffer overflow in GLOBAL memory access");
                }
            }
        }
        
        if (!found) {
            PTX_DEBUG_MEM("Accessing unallocated GLOBAL memory: ptr=%p, size=%zu", host_ptr, size);
            throw std::runtime_error("Accessing unallocated GLOBAL memory");
        }

        PTX_DEBUG_MEM("GLOBAL memory %s: ptr=%p, host_offset=%zu, size=%zu", 
            is_write ? "WRITE" : "READ", host_ptr, static_cast<uint8_t*>(host_ptr) - allocator_->get_pool(), size);

        MemoryAccess req{.space = space,
                         .address = static_cast<uint8_t*>(host_ptr) - allocator_->get_pool(),
                         .size = size,
                         .is_write = is_write,
                         .data = data};
        memory_interface_->access(req);
        break;
    }

    case MemorySpace::PARAM: {
        // 对于PARAM空间，需要检查host_ptr是否在已分配的范围内
        // 而不是精确匹配起始地址
        bool found = false;
        SimpleMemoryAllocator::Allocation *alloc = nullptr;
        
        // 遍历param_allocations_查找包含host_ptr的区间
        for (auto& pair : param_allocations_) {
            auto allocation = allocator_->get_allocation(pair.second.offset);
            if (!allocation) continue;
            
            uint8_t* start_addr = allocator_->get_pool() + allocation->offset;
            uint8_t* end_addr = start_addr + allocation->size;
            uint8_t* access_addr = static_cast<uint8_t*>(host_ptr);
            
            if (access_addr >= start_addr && access_addr < end_addr) {
                // 确保访问的范围没有超出分配的范围
                if (access_addr + size <= end_addr) {
                    alloc = allocation;
                    found = true;
                    break;
                } else {
                    PTX_DEBUG_MEM("Buffer overflow in PARAM memory access: ptr=%p, size=%zu, access range=[%p, %p], allocated range=[%p, %p]", 
                        host_ptr, size, access_addr, access_addr + size, start_addr, end_addr);
                    throw std::runtime_error("Buffer overflow in PARAM memory access");
                }
            }
        }
        
        if (!found) {
            PTX_DEBUG_MEM("Accessing unallocated PARAM memory: ptr=%p, size=%zu", host_ptr, size);
            throw std::runtime_error("Accessing unallocated PARAM memory");
        }

        PTX_DEBUG_MEM("PARAM memory %s: ptr=%p, size=%zu", 
            is_write ? "WRITE" : "READ", host_ptr, size);

        // 对于PARAM空间，直接进行内存拷贝，不需要通过MemoryInterface
        if (is_write) {
            std::memcpy(host_ptr, data, size);
        } else {
            std::memcpy(data, host_ptr, size);
        }
        break;
    }

    case MemorySpace::SHARED: {
        // 对于SHARED空间，检查host_ptr是否在共享内存池范围内
        uint8_t* access_addr = static_cast<uint8_t*>(host_ptr);
        uint8_t* pool_start = shared_pool_;
        uint8_t* pool_end = shared_pool_ + SHARED_SIZE;
        
        if (access_addr >= pool_start && access_addr + size <= pool_end) {
            // 验证访问的地址是否在已分配的共享内存块中
            bool found = false;
            for (const auto& pair : shared_allocations_) {
                uint8_t* block_start = shared_pool_ + pair.second.offset;
                uint8_t* block_end = block_start + pair.second.size;
                
                // 检查访问范围是否在分配的块内
                if (access_addr >= block_start && access_addr + size <= block_end) {
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                PTX_DEBUG_MEM("Accessing unallocated SHARED memory: ptr=%p, size=%zu", host_ptr, size);
                throw std::runtime_error("Accessing unallocated SHARED memory");
            }

            PTX_DEBUG_MEM("SHARED memory %s: ptr=%p, size=%zu", 
                is_write ? "WRITE" : "READ", host_ptr, size);

            // 对于SHARED空间，直接进行内存拷贝
            if (is_write) {
                std::memcpy(host_ptr, data, size);
            } else {
                std::memcpy(data, host_ptr, size);
            }
        } else {
            PTX_DEBUG_MEM("SHARED memory access out of bounds: ptr=%p, size=%zu", host_ptr, size);
            throw std::runtime_error("SHARED memory access out of bounds");
        }
        break;
    }

    default: {
        // 其他地址空间暂时沿用原来的处理方式
        auto it = allocations_.find(reinterpret_cast<uint64_t>(host_ptr));
        if (it == allocations_.end()) {
            PTX_DEBUG_MEM("Accessing unallocated memory: ptr=%p, size=%zu", host_ptr, size);
            throw std::runtime_error("Accessing unallocated memory");
        }

        const auto &alloc = it->second;
        if (size > alloc.size) {
            PTX_DEBUG_MEM("Buffer overflow in memory access: ptr=%p, requested_size=%zu, allocated_size=%zu", 
                host_ptr, size, alloc.size);
            throw std::runtime_error("Buffer overflow in memory access");
        }

        PTX_DEBUG_MEM("MEMORY %s: ptr=%p, host_offset=%zu, size=%zu", 
            is_write ? "WRITE" : "READ", host_ptr, static_cast<uint8_t*>(host_ptr) - allocator_->get_pool(), size);

        MemoryAccess req{.space = space,
                         .address = static_cast<uint8_t*>(host_ptr) - allocator_->get_pool(),
                         .size = size,
                         .is_write = is_write,
                         .data = data};
        memory_interface_->access(req);
        break;
    }
    }
}