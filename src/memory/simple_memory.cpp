// memory/simple_memory.cpp
#include "memory/simple_memory.h"
#include "utils/logger.h"
#include <stdexcept>
#include <sys/mman.h>

SimpleMemory::SimpleMemory(size_t global_size)
    : global_size_(global_size), global_base_(nullptr) {

    // 分配全局内存池
    global_base_ = static_cast<uint8_t *>(
        mmap(nullptr, global_size_, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (global_base_ == MAP_FAILED) {
        throw std::runtime_error("Failed to mmap global memory pool");
    }
}

SimpleMemory::~SimpleMemory() {
    if (global_base_ && global_base_ != MAP_FAILED) {
        munmap(global_base_, global_size_);
    }
}

void SimpleMemory::direct_access(uint64_t address, void *data, size_t size,
                                 bool is_write) {
    // 确保访问不超过全局内存池大小
    if (address + size > global_size_) {
        throw std::runtime_error(
            "Memory access out of bounds: addr=" + std::to_string(address) +
            ", size=" + std::to_string(size) +
            ", max_size=" + std::to_string(global_size_));
    }

    // 执行访问
    if (is_write) {
        std::memcpy(global_base_ + address, data, size);
    } else {
        std::memcpy(data, global_base_ + address, size);
    }
}

// void SimpleMemory::access(const MemoryAccess &req) {
//     uint8_t *base = nullptr;
//     size_t max_size = 0;

//     base = global_base_;
//     max_size = global_size_;

//     // 边界检查
//     if (req.address + req.size > max_size) {
//         throw std::runtime_error("Memory access out of bounds: space=" +
//                                  std::to_string(static_cast<int>(req.space))
//                                  +
//                                  ", addr=" + std::to_string(req.address) +
//                                  ", size=" + std::to_string(req.size) +
//                                  ", max_size=" + std::to_string(max_size));
//     }

//     // 执行访问
//     if (req.is_write) {
//         std::memcpy(base + req.address, req.data, req.size);
//     } else {
//         std::memcpy(req.data, base + req.address, req.size);
//     }
// }