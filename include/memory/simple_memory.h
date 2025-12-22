// memory/simple_memory.h
#ifndef SIMPLE_MEMORY_H
#define SIMPLE_MEMORY_H

#include "memory_interface.h"
#include <memory>

class SimpleMemory : public MemoryInterface {
public:
    SimpleMemory(uint8_t *global_base, size_t global_size, uint8_t *shared_base,
                 size_t shared_size);
    ~SimpleMemory() override = default;

    void access(const MemoryAccess &req) override;

private:
    uint8_t *global_base_;
    size_t global_size_;
    uint8_t *shared_base_;
    size_t shared_size_;
};

#endif // SIMPLE_MEMORY_H