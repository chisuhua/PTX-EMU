#ifndef SIMPLE_MEMORY_ALLOCATOR_H
#define SIMPLE_MEMORY_ALLOCATOR_H

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <sys/mman.h>
#include <vector>

class SimpleMemoryAllocator {
public:
    static constexpr size_t GLOBAL_SIZE = 4ULL << 30; // 4GB

    struct Allocation {
        size_t offset;
        size_t size;
    };

    // 不再自动创建内存池
    SimpleMemoryAllocator() = default;
    ~SimpleMemoryAllocator() = default;

    // 初始化时传入内存池和大小
    void init(size_t size);

    // 分配指定大小的内存，返回在内存池中的偏移量
    size_t allocate(size_t size);

    // 释放指定偏移量的内存
    void deallocate(size_t offset);

    // 获取内存池大小
    size_t get_pool_size() const { return pool_size_; }

    // 获取已分配内存的信息
    Allocation *get_allocation(size_t offset);

private:
    size_t pool_size_ = 0;

    // 已分配的内存块，按偏移量排序
    std::map<size_t, Allocation> allocated_blocks_;

    // 空闲内存块，按大小排序（first是大小，second是偏移量）
    std::multimap<size_t, size_t> free_blocks_;

    mutable std::mutex mutex_;

    // 用于快速查找已分配块
    size_t last_allocated_end_ = 0; // 记录最后分配的块的结束位置，用于快速分配
};

#endif // SIMPLE_MEMORY_ALLOCATOR_H