// memory/simple_memory.h
#ifndef SIMPLE_MEMORY_H
#define SIMPLE_MEMORY_H

#include <memory>

class SimpleMemory {
public:
    SimpleMemory(size_t global_size);
    ~SimpleMemory();

    // 直接访问内存池数据
    void direct_access(uint64_t address, void *data, size_t size,
                       bool is_write);

    // 获取内存池指针
    uint8_t *get_global_pool() const { return global_base_; }
    size_t get_global_size() const { return global_size_; }

private:
    uint8_t *global_base_;
    size_t global_size_;
};

#endif // SIMPLE_MEMORY_H