// memory/simple_memory.h
#ifndef SIMPLE_MEMORY_H
#define SIMPLE_MEMORY_H

#include <cstddef>
#include <cstdint>
#include <memory>

class SimpleMemory {
public:
    SimpleMemory(size_t global_size);
    ~SimpleMemory();

    bool validate_offset(uint64_t offset, size_t size) const;
    void direct_access(uint64_t address, void *data, size_t size,
                       bool is_write);

    uint8_t *get_global_pool() const { return global_base_; }
    size_t get_global_size() const { return global_size_; }

private:
    uint8_t *global_base_;
    size_t global_size_;
};

#endif