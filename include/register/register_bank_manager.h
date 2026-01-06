#ifndef REGISTER_BANK_MANAGER_H
#define REGISTER_BANK_MANAGER_H

#include "ptx_ir/ptx_types.h"
#include "ptxsim/register_analyzer.h"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// 寄存器银行管理器，为整个CTA提供统一的寄存器存储
class RegisterBankManager {
public:
    RegisterBankManager(int max_warps, int threads_per_warp);
    virtual ~RegisterBankManager() = default;

    // 为所有线程预分配指定名称的寄存器
    bool create_register(const std::string &name, size_t size);

    // 为指定warp和lane获取寄存器的物理地址
    void *get_register(const std::string &name, int warp_id, int lane_id);

    // 为所有线程预分配寄存器
    void preallocate_registers(const std::vector<RegisterInfo> &registers);

    // 重置所有寄存器
    void reset();

private:
    int max_warps_;
    int threads_per_warp_;
    int total_threads_;

    // 存储寄存器信息
    struct RegisterDesc {
        std::string name;
        size_t size;
        // 存储格式为 [warp_id][lane_id][register_data]
        std::vector<std::vector<std::vector<uint8_t>>> data_storage;
    };

    // 存储所有寄存器描述
    std::unordered_map<std::string, RegisterDesc> register_descriptions_;

    // 线程安全锁
    std::mutex mutex_;
};

#endif // REGISTER_BANK_MANAGER_H