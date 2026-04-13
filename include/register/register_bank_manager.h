#ifndef REGISTER_BANK_MANAGER_H
#define REGISTER_BANK_MANAGER_H

#include "ptx_ir/ptx_types.h"
#include "ptxsim/register_analyzer.h"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// 寄存器银行管理器，为整个CTA提供统一的寄存器存储
enum class RegisterStorageMode {
    PER_LANE_BYTES,
    WARP_BITMASK
};

class RegisterBankManager {
public:
    RegisterBankManager(int max_warps, int threads_per_warp);
    virtual ~RegisterBankManager() = default;

    bool create_register(const std::string &name, size_t size);

    void *get_register(const std::string &name, int warp_id, int lane_id);

    void preallocate_registers(const std::vector<RegisterInfo> &registers);

    void reset();

    uint32_t get_predicate_mask(const std::string &name, int warp_id);
    void set_predicate_mask(const std::string &name, int warp_id, uint32_t mask);
    bool get_predicate_bit(const std::string &name, int warp_id, int lane_id);
    void set_predicate_bit(const std::string &name, int warp_id, int lane_id, bool value);

private:
    int max_warps_;
    int threads_per_warp_;
    int total_threads_;

    struct RegisterDesc {
        std::string name;
        size_t size;
        RegisterStorageMode mode;
        std::vector<std::vector<std::vector<uint8_t>>> data_storage;
        std::vector<uint32_t> predicate_mask_storage;
    };

    static bool is_predicate_name(const std::string &name);

    std::unordered_map<std::string, RegisterDesc> register_descriptions_;
    std::mutex mutex_;
};

#endif // REGISTER_BANK_MANAGER_H