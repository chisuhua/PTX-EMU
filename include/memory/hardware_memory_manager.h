#ifndef HARDWARE_MEMORY_MANAGER_H
#define HARDWARE_MEMORY_MANAGER_H

#include "memory/memory_interface.h"
#include <mutex>

class SimpleMemory; // 前向声明

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

private:
    HardwareMemoryManager();
    ~HardwareMemoryManager();

    SimpleMemory *simple_memory_ = nullptr;
    mutable std::mutex mutex_;
    
    // 内存访问统计
    size_t read_count_ = 0;
    size_t write_count_ = 0;
};

#endif // HARDWARE_MEMORY_MANAGER_H