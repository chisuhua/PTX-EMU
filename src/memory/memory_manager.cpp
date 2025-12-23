// memory/memory_manager.cpp
#include "memory/memory_manager.h"
#include <cstring>
#include <stdexcept>
#include <unistd.h>

MemoryManager &MemoryManager::instance() {
    static MemoryManager inst;
    return inst;
}

MemoryManager::MemoryManager() {
    // 分配 GLOBAL 内存池（mmap 虚拟内存）
    global_pool_ = static_cast<uint8_t *>(
        mmap(nullptr, GLOBAL_SIZE, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (global_pool_ == MAP_FAILED) {
        throw std::runtime_error("Failed to mmap global memory pool");
    }

    // 分配 SHARED 内存池（小，直接 malloc）
    shared_pool_ = new uint8_t[SHARED_SIZE];
    std::memset(shared_pool_, 0, SHARED_SIZE);
}

MemoryManager::~MemoryManager() {
    if (global_pool_ && global_pool_ != MAP_FAILED) {
        munmap(global_pool_, GLOBAL_SIZE);
    }
    delete[] shared_pool_;
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
    if (global_offset_ + size > GLOBAL_SIZE) {
        return nullptr;
    }

    void *host_ptr = global_pool_ + global_offset_;
    allocations_[reinterpret_cast<uint64_t>(host_ptr)] = {global_offset_, size};
    global_offset_ += size;
    return host_ptr;
}

mycudaError_t MemoryManager::free(void *dev_ptr) {
    if (!dev_ptr)
        return Success;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(reinterpret_cast<uint64_t>(dev_ptr));
    if (it != allocations_.end()) {
        allocations_.erase(it);
        return Success;
    }
    return ErrorInvalidValue;
}

void *MemoryManager::malloc_param(size_t size) {
    if (size == 0)
        return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);
    
    // 为PARAM空间分配内存，从global_pool_中分配
    if (param_offset_ + size > GLOBAL_SIZE) {
        return nullptr;  // 检查是否有足够的空间
    }

    void *param_ptr = global_pool_ + param_offset_;
    param_allocations_[reinterpret_cast<uint64_t>(param_ptr)] = {param_offset_, size};
    param_offset_ += size;
    return param_ptr;
}

void MemoryManager::free_param(void *param_ptr) {
    if (!param_ptr)
        return;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = param_allocations_.find(reinterpret_cast<uint64_t>(param_ptr));
    if (it != param_allocations_.end()) {
        // 简单地从映射中删除记录
        param_allocations_.erase(it);
        
        // 注意：这里我们没有真正"释放"内存，因为使用的是连续分配
        // 如果需要更复杂的内存管理，可以考虑实现内存池或垃圾回收机制
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
        auto it = allocations_.find(reinterpret_cast<uint64_t>(host_ptr));
        if (it == allocations_.end()) {
            throw std::runtime_error("Accessing unallocated GLOBAL memory");
        }

        const auto &alloc = it->second;
        if (size > alloc.size) {
            throw std::runtime_error("Buffer overflow in GLOBAL memory access");
        }

        MemoryAccess req{.space = space,
                         .address = alloc.offset,
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
        Allocation *alloc = nullptr;
        
        // 遍历param_allocations_查找包含host_ptr的区间
        for (auto& pair : param_allocations_) {
            Allocation& allocation = pair.second;
            uint8_t* start_addr = global_pool_ + allocation.offset;
            uint8_t* end_addr = start_addr + allocation.size;
            uint8_t* access_addr = static_cast<uint8_t*>(host_ptr);
            
            if (access_addr >= start_addr && access_addr < end_addr) {
                // 确保访问的范围没有超出分配的范围
                if (access_addr + size <= end_addr) {
                    alloc = &allocation;
                    found = true;
                    break;
                } else {
                    throw std::runtime_error("Buffer overflow in PARAM memory access");
                }
            }
        }
        
        if (!found) {
            throw std::runtime_error("Accessing unallocated PARAM memory");
        }

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
            throw std::runtime_error("Accessing unallocated memory");
        }

        const auto &alloc = it->second;
        if (size > alloc.size) {
            throw std::runtime_error("Buffer overflow in memory access");
        }

        MemoryAccess req{.space = space,
                         .address = alloc.offset,
                         .size = size,
                         .is_write = is_write,
                         .data = data};
        memory_interface_->access(req);
        break;
    }
    }
}
