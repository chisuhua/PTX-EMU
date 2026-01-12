#include "memory/hardware_memory_manager.h"
#include "memory/simple_memory.h"
#include "utils/logger.h"
#include <cstring>
#include <stdexcept>

HardwareMemoryManager &HardwareMemoryManager::instance() {
    static HardwareMemoryManager inst;
    return inst;
}

HardwareMemoryManager::HardwareMemoryManager() = default;

HardwareMemoryManager::~HardwareMemoryManager() = default;

void HardwareMemoryManager::set_simple_memory(SimpleMemory *simple_memory) {
    std::lock_guard<std::mutex> lock(mutex_);
    simple_memory_ = simple_memory;
}

void HardwareMemoryManager::access(const MemoryAccess &req) {
    switch (req.space) {
    case MemorySpace::SHARED:
        // 对于共享内存访问，直接进行内存操作（因为共享内存地址已经是真实地址）
        if (req.is_write) {
            std::memcpy(reinterpret_cast<void *>(req.address), req.data,
                        req.size);
        } else {
            std::memcpy(req.data, reinterpret_cast<void *>(req.address),
                        req.size);
        }
        break;

    case MemorySpace::GLOBAL:
    case MemorySpace::PARAM:
    case MemorySpace::LOCAL:
    case MemorySpace::CONST:
        // 对于其他内存空间，通过SimpleMemory访问（使用偏移地址）
        if (simple_memory_) {
            simple_memory_->direct_access(req.address, req.data, req.size,
                                          req.is_write);
        } else {
            throw std::runtime_error(
                "SimpleMemory not set in HardwareMemoryManager");
        }
        break;

    default:
        throw std::runtime_error(
            "Unsupported memory space in HardwareMemoryManager::access");
    }

    // 更新统计信息
    if (req.is_write) {
        write_count_++;
    } else {
        read_count_++;
    }

    PTX_DEBUG_MEM("%s to memory: addr=0x%lx, size=%zu, space=%d",
                  req.is_write ? "Write" : "Read", req.address, req.size,
                  static_cast<int>(req.space));
}

void HardwareMemoryManager::access(void *dev_ptr, void *data, size_t size,
                                   bool is_write, MemorySpace space) {
    if (!dev_ptr || !data || size == 0) {
        throw std::invalid_argument("Invalid memory access arguments");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 根据地址空间类型处理访问
    switch (space) {
    case MemorySpace::SHARED:
        // 对于共享内存访问，直接进行内存操作（因为共享内存地址已经是真实地址）
        if (is_write) {
            std::memcpy(dev_ptr, data, size);
        } else {
            std::memcpy(data, dev_ptr, size);
        }
        break;

    case MemorySpace::GLOBAL:
    case MemorySpace::PARAM:
    case MemorySpace::LOCAL:
    case MemorySpace::CONST: {
        // 对于其他内存空间，获取偏移量并使用SimpleMemory访问
        simple_memory_->direct_access((uint64_t)dev_ptr, data, size, is_write);
        break;
    }

    default: {
        throw std::runtime_error(
            "Unsupported memory space in HardwareMemoryManager::access");
    }
    }
}