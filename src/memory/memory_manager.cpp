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

void MemoryManager::access(void *host_ptr, void *data, size_t size,
                           bool is_write) {
    if (!host_ptr || !data || size == 0) {
        throw std::invalid_argument("Invalid memory access arguments");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!memory_interface_) {
        throw std::runtime_error("MemoryInterface not initialized");
    }

    auto it = allocations_.find(reinterpret_cast<uint64_t>(host_ptr));
    if (it == allocations_.end()) {
        throw std::runtime_error("Accessing unallocated memory");
    }

    const auto &alloc = it->second;
    if (size > alloc.size) {
        throw std::runtime_error("Buffer overflow in memory access");
    }

    MemoryAccess req{.space = MemorySpace::GLOBAL,
                     .address = alloc.offset,
                     .size = size,
                     .is_write = is_write,
                     .data = data};
    memory_interface_->access(req);
}