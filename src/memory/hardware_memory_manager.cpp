#include "memory/hardware_memory_manager.h"
#include "memory/simple_memory.h"
#include "ptxsim/ptx_exceptions.h"
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
    uint64_t addr = req.address;

    const char* region_name = nullptr;
    switch (req.space) {
    case MemorySpace::SHARED:    region_name = "shared"; break;
    case MemorySpace::GLOBAL:    region_name = "global"; break;
    case MemorySpace::LOCAL:     region_name = "local"; break;
    case MemorySpace::CONST:    region_name = "constant"; break;
    case MemorySpace::PARAM:    region_name = "param"; break;
    default: break;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (region_name) {
        auto it = regions_.find(region_name);
        if (it != regions_.end()) {
            const MemoryRegion& region = it->second;
            if (!region.contains(addr, req.size)) {
                throw InvalidMemoryAccessException(
                    addr, req.size, "out of bounds",
                    "Access at 0x" + std::to_string(addr) +
                    " exceeds region " + region_name +
                    " [base=0x" + std::to_string(region.base_address) +
                    ", size=" + std::to_string(region.size) + "]");
            }
            if (req.is_write && !region.is_writable) {
                throw InvalidMemoryAccessException(
                    addr, req.size, "write to read-only region",
                    "Region " + std::string(region_name) + " is not writable");
            }
        }
    }

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

    uint64_t addr = reinterpret_cast<uint64_t>(dev_ptr);

    const char* region_name = nullptr;
    switch (space) {
    case MemorySpace::SHARED:    region_name = "shared"; break;
    case MemorySpace::GLOBAL:    region_name = "global"; break;
    case MemorySpace::LOCAL:     region_name = "local"; break;
    case MemorySpace::CONST:    region_name = "constant"; break;
    case MemorySpace::PARAM:    region_name = "param"; break;
    default: break;
    }
    if (region_name) {
        auto it = regions_.find(region_name);
        if (it != regions_.end()) {
            const MemoryRegion& region = it->second;
            if (!region.contains(addr, size)) {
                throw InvalidMemoryAccessException(
                    addr, size, "out of bounds",
                    "Access at 0x" + std::to_string(addr) +
                    " exceeds region " + region_name +
                    " [base=0x" + std::to_string(region.base_address) +
                    ", size=" + std::to_string(region.size) + "]");
            }
            if (is_write && !region.is_writable) {
                throw InvalidMemoryAccessException(
                    addr, size, "write to read-only region",
                    "Region " + std::string(region_name) + " is not writable");
            }
        }
    }

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
        // LOCAL 是 per-thread 真实指针；若 simple_memory_ 未注册
        // （如 setup_block 单元测试），直接 memcpy 避免 nullptr 解引用
        if (space == MemorySpace::LOCAL && simple_memory_ == nullptr) {
            if (is_write) {
                std::memcpy(dev_ptr, data, size);
            } else {
                std::memcpy(data, dev_ptr, size);
            }
            break;
        }
        simple_memory_->direct_access((uint64_t)dev_ptr, data, size, is_write);
        break;
    }

    default: {
        throw std::runtime_error(
            "Unsupported memory space in HardwareMemoryManager::access");
    }
    }
}

bool HardwareMemoryManager::register_region(const MemoryRegion& region) {
    std::lock_guard<std::mutex> lock(mutex_);
    regions_[region.name] = region;
    return true;
}

bool HardwareMemoryManager::unregister_region(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return regions_.erase(name) > 0;
}

const MemoryRegion* HardwareMemoryManager::get_region(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = regions_.find(name);
    if (it != regions_.end()) {
        return &it->second;
    }
    return nullptr;
}

const std::unordered_map<std::string, MemoryRegion>&
HardwareMemoryManager::get_all_regions() const {
    return regions_;
}