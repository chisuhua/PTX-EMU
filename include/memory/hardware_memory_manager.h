#ifndef HARDWARE_MEMORY_MANAGER_H
#define HARDWARE_MEMORY_MANAGER_H

#include "memory/memory_interface.h"
#include <mutex>
#include <string>
#include <unordered_map>

class SimpleMemory;

struct MemoryRegion {
    std::string name;
    uint64_t base_address;
    uint64_t size;
    bool is_writable;
    bool is_executable;

    bool contains(uint64_t addr, size_t access_size = 1) const {
        return addr >= base_address &&
               (addr + access_size) <= (base_address + size);
    }
};

class HardwareMemoryManager : public MemoryInterface {
public:
    static HardwareMemoryManager &instance();

    // 禁用拷贝
    HardwareMemoryManager(const HardwareMemoryManager &) = delete;
    HardwareMemoryManager &operator=(const HardwareMemoryManager &) = delete;

    // 设置SimpleMemory实例
    void set_simple_memory(SimpleMemory *simple_memory);

    // 实现 MemoryInterface
    void access(const MemoryAccess &req) override;

    // 内存访问接口 - 用于 ptx 指令执行
    void access(void *host_ptr, void *data, size_t size, bool is_write, MemorySpace space);

    bool register_region(const MemoryRegion& region);
    bool unregister_region(const std::string& name);
    const MemoryRegion* get_region(const std::string& name) const;
    const std::unordered_map<std::string, MemoryRegion>& get_all_regions() const;

private:
    HardwareMemoryManager();
    ~HardwareMemoryManager();

    SimpleMemory *simple_memory_ = nullptr;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, MemoryRegion> regions_;
    
    // 内存访问统计
    size_t read_count_ = 0;
    size_t write_count_ = 0;
};

#endif // HARDWARE_MEMORY_MANAGER_H