#ifndef DRIVER_MEMORY_MANAGER_H
#define DRIVER_MEMORY_MANAGER_H

#include "cudart/simple_memory_allocator.h"
#include "memory/simple_memory.h"  // 包含SimpleMemory的完整定义
#include <mutex>
#include <unordered_map>

enum mycudaError_t {
    Success = 0,
    ErrorMemoryAllocation = 2,
    ErrorInvalidValue = 11,
    ErrorInitializationError = 3
};

class CudaDriver {
public:
    static CudaDriver &instance();

    // 禁用拷贝
    CudaDriver(const CudaDriver &) = delete;
    CudaDriver &operator=(const CudaDriver &) = delete;

    // 设置SimpleMemory实例
    void set_simple_memory(SimpleMemory *simple_memory) {
        simple_memory_ = simple_memory;
    };

    // CUDA 内存分配 - 用于 cudart_sim 和 ptx_interpreter
    void *malloc(size_t size);
    void *malloc_managed(size_t size);
    mycudaError_t free(void *dev_ptr);

    // PARAM空间内存分配和释放 - 用于 cudart_sim 和 ptx_interpreter
    // void *malloc_param(size_t size);
    // void free_param(void *param_ptr);

    // 获取内存池指针（供 cudart_sim 使用）
    uint8_t *get_global_pool() const;

    static constexpr size_t GLOBAL_SIZE = 4ULL << 30; // 4GB

private:
    CudaDriver();
    ~CudaDriver();

    SimpleMemory *simple_memory_ = nullptr;
    SimpleMemoryAllocator *allocator_ = nullptr;

    mutable std::mutex mutex_;

    // 主机指针 → {偏移, 大小}
    struct Allocation {
        size_t offset;
        size_t size;
    };
    std::unordered_map<uint64_t, Allocation> allocations_;

    // PARAM空间管理
    std::unordered_map<uint64_t, Allocation> param_allocations_;
};

#endif // DRIVER_MEMORY_MANAGER_H