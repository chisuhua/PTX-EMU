#include "cudart/cuda_driver.h"
#include "cudart/simple_memory_allocator.h"
#include "utils/logger.h" // 添加日志头文件
#include <cstring>
#include <mutex> // 添加mutex头文件
#include <stdexcept>
#include <unistd.h>
#include <unordered_map> // 添加unordered_map头文件

// namespace {
// SimpleMemoryAllocator *get_global_allocator() {
//     static SimpleMemoryAllocator allocator;
//     return &allocator;
// }
// } // namespace

CudaDriver &CudaDriver::instance() {
    static CudaDriver inst;
    return inst;
}

CudaDriver::CudaDriver() {
    // 使用全局内存分配器
    allocator_ =
        std::make_unique<SimpleMemoryAllocator>(); // get_global_allocator();
}

CudaDriver::~CudaDriver() {
    // 不需要释放allocator_，因为它是一个全局单例
}

void *CudaDriver::malloc(size_t size) {
    if (size == 0)
        return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);

    size_t offset = allocator_->allocate(size);
    if (offset == static_cast<size_t>(-1)) {
        return nullptr;
    }

    // 获取全局内存池基址
    uint8_t *global_pool = reinterpret_cast<uint8_t *>(get_global_pool());
    void *dev_ptr = global_pool + offset;

    allocations_[reinterpret_cast<uint64_t>(dev_ptr)] = {offset, size};

    PTX_DEBUG_MEM("GLOBAL memory allocated: devptr=%p, size=%zu", dev_ptr,
                  size);
    return dev_ptr;
}

void *CudaDriver::malloc_managed(size_t size) {
    if (size == 0)
        return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);

    size_t offset = allocator_->allocate(size);
    if (offset == static_cast<size_t>(-1)) {
        return nullptr;
    }

    // 获取全局内存池基址
    uint8_t *global_pool = reinterpret_cast<uint8_t *>(get_global_pool());
    void *host_ptr = global_pool + offset;
    allocations_[reinterpret_cast<uint64_t>(host_ptr)] = {offset, size};

    PTX_DEBUG_MEM("MANAGED memory allocated: ptr=%p, offset=%zu, size=%zu",
                  host_ptr, offset, size);
    return host_ptr;
}

mycudaError_t CudaDriver::free(void *dev_ptr) {
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

uint8_t *CudaDriver::get_global_pool() const {
    if (simple_memory_) {
        return simple_memory_->get_global_pool();
    }
    // 如果没有设置simple_memory_，返回nullptr或抛出异常
    return nullptr;
}

size_t CudaDriver::get_global_size() const { return global_size_; }

// void *CudaDriver::malloc_param(size_t size) {
//     if (size == 0)
//         return nullptr;

//     std::lock_guard<std::mutex> lock(mutex_);

//     size_t offset = allocator_->allocate(size);
//     if (offset == static_cast<size_t>(-1)) {
//         return nullptr;
//     }

//     void *param_ptr = allocator_->get_pool() + offset;
//     param_allocations_[reinterpret_cast<uint64_t>(param_ptr)] = {offset,
//     size};

//     PTX_DEBUG_MEM("PARAM memory allocated: ptr=%p, offset=%zu, size=%zu",
//                   param_ptr, offset, size);
//     return param_ptr;
// }

// void CudaDriver::free_param(void *param_ptr) {
//     if (!param_ptr)
//         return;

//     std::lock_guard<std::mutex> lock(mutex_);
//     auto it = param_allocations_.find(reinterpret_cast<uint64_t>(param_ptr));
//     if (it != param_allocations_.end()) {
//         allocator_->deallocate(it->second.offset);
//         param_allocations_.erase(it);
//     }
// }