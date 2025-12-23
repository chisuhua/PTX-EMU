// memory/memory_manager.h
#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include "memory_interface.h"
#include <cstdint>
#include <mutex>
#include <sys/mman.h>
#include <unordered_map>

enum mycudaError_t {
    Success = 0,
    ErrorMemoryAllocation = 2,
    ErrorInvalidValue = 11
};

class MemoryManager {
public:
    static MemoryManager &instance();

    // 禁用拷贝
    MemoryManager(const MemoryManager &) = delete;
    MemoryManager &operator=(const MemoryManager &) = delete;

    // 初始化 MemoryInterface（必须在 cudaMalloc 前调用）
    void set_memory_interface(MemoryInterface *mem_if);

    // CUDA 内存分配
    void *malloc(size_t size);
    void *malloc_managed(size_t size);
    mycudaError_t free(void *dev_ptr);

    // PARAM空间内存分配和释放
    void *malloc_param(size_t size);
    void free_param(void *param_ptr);

    // PTX 指令内存访问
    void access(void *host_ptr, void *data, size_t size, bool is_write);
    void access(void *host_ptr, void *data, size_t size, bool is_write, MemorySpace space);

    // 获取内存池指针（供 SimpleMemory 使用）
    uint8_t *get_global_pool() const { return global_pool_; }
    uint8_t *get_shared_pool() const { return shared_pool_; }

    static constexpr size_t GLOBAL_SIZE = 4ULL << 30; // 4GB
    static constexpr size_t SHARED_SIZE = 64 * 1024;  // 64KB

private:
    MemoryManager();
    ~MemoryManager();

    // 唯一内存池（mmap 虚拟内存）
    uint8_t *global_pool_ = nullptr;
    uint8_t *shared_pool_ = nullptr;

    size_t global_offset_ = 0;
    size_t param_offset_ = 0;  // 为PARAM空间添加偏移量
    mutable std::mutex mutex_;

    // 主机指针 → {偏移, 大小}
    struct Allocation {
        size_t offset;
        size_t size;
    };
    std::unordered_map<uint64_t, Allocation> allocations_;
    
    // PARAM空间管理
    std::unordered_map<uint64_t, Allocation> param_allocations_;

    MemoryInterface *memory_interface_ = nullptr;
};

#endif // MEMORY_MANAGER_H