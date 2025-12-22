// memory/simple_memory.cpp
#include "memory/simple_memory.h"
#include <cstring>
#include <stdexcept>

SimpleMemory::SimpleMemory(uint8_t *global_base, size_t global_size,
                           uint8_t *shared_base, size_t shared_size)
    : global_base_(global_base), global_size_(global_size),
      shared_base_(shared_base), shared_size_(shared_size) {}

void SimpleMemory::access(const MemoryAccess &req) {
    uint8_t *base = nullptr;
    size_t max_size = 0;

    switch (req.space) {
    case MemorySpace::SHARED:
        base = shared_base_;
        max_size = shared_size_;
        break;
    case MemorySpace::GLOBAL:
        base = global_base_;
        max_size = global_size_;
        break;
    default:
        throw std::runtime_error("Unsupported memory space");
    }

    // 边界检查
    if (req.address + req.size > max_size) {
        throw std::runtime_error("Memory access out of bounds: space=" +
                                 std::to_string(static_cast<int>(req.space)) +
                                 ", addr=" + std::to_string(req.address) +
                                 ", size=" + std::to_string(req.size) +
                                 ", max_size=" + std::to_string(max_size));
    }

    // 执行访问
    if (req.is_write) {
        std::memcpy(base + req.address, req.data, req.size);
    } else {
        std::memcpy(req.data, base + req.address, req.size);
    }
}