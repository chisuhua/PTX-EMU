// memory/memory_manager.cpp
#include "memory/memory_manager.h"
#include "memory/simple_memory_allocator.h"
#include <cstring>
#include <stdexcept>
#include <unistd.h>
#include "utils/logger.h"

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
}

MemoryManager::~MemoryManager() {
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