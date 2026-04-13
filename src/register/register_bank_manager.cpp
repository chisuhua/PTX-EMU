#include "register/register_bank_manager.h"
#include <iostream>

RegisterBankManager::RegisterBankManager(int max_warps, int threads_per_warp)
    : max_warps_(max_warps), threads_per_warp_(threads_per_warp),
      total_threads_(max_warps * threads_per_warp) {
    register_descriptions_.clear();
}

bool RegisterBankManager::create_register(const std::string &name,
                                          size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (register_descriptions_.find(name) != register_descriptions_.end()) {
        return false;
    }

    RegisterDesc desc;
    desc.name = name;
    desc.size = size;
    desc.mode = RegisterStorageMode::PER_LANE_BYTES;
    desc.data_storage.resize(max_warps_);
    for (int w = 0; w < max_warps_; w++) {
        desc.data_storage[w].resize(threads_per_warp_);
        for (int l = 0; l < threads_per_warp_; l++) {
            desc.data_storage[w][l].resize(size, 0);
        }
    }

    register_descriptions_[name] = desc;
    return true;
}

void *RegisterBankManager::get_register(const std::string &name, int warp_id,
                                         int lane_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (warp_id >= max_warps_ || warp_id < 0) return nullptr;
    if (lane_id >= threads_per_warp_ || lane_id < 0) return nullptr;

    auto it = register_descriptions_.find(name);
    if (it == register_descriptions_.end()) return nullptr;

    auto &desc = it->second;
    if (desc.mode == RegisterStorageMode::WARP_BITMASK) {
        return nullptr;
    }

    if (warp_id >= (int)desc.data_storage.size()) return nullptr;
    if (lane_id >= (int)desc.data_storage[warp_id].size()) return nullptr;

    return desc.data_storage[warp_id][lane_id].data();
}

uint32_t RegisterBankManager::get_predicate_mask(const std::string &name, int warp_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = register_descriptions_.find(name);
    if (it == register_descriptions_.end()) return 0;
    if (it->second.mode != RegisterStorageMode::WARP_BITMASK) return 0;
    if (warp_id < 0 || warp_id >= (int)it->second.predicate_mask_storage.size()) return 0;
    return it->second.predicate_mask_storage[warp_id];
}

void RegisterBankManager::set_predicate_mask(const std::string &name, int warp_id, uint32_t mask) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = register_descriptions_.find(name);
    if (it == register_descriptions_.end()) return;
    if (it->second.mode != RegisterStorageMode::WARP_BITMASK) return;
    if (warp_id < 0 || warp_id >= (int)it->second.predicate_mask_storage.size()) return;
    it->second.predicate_mask_storage[warp_id] = mask;
}

bool RegisterBankManager::get_predicate_bit(const std::string &name, int warp_id, int lane_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = register_descriptions_.find(name);
    if (it == register_descriptions_.end()) return false;
    if (it->second.mode != RegisterStorageMode::WARP_BITMASK) return false;
    if (warp_id < 0 || warp_id >= (int)it->second.predicate_mask_storage.size()) return false;
    return (it->second.predicate_mask_storage[warp_id] >> lane_id) & 1;
}

void RegisterBankManager::set_predicate_bit(const std::string &name, int warp_id, int lane_id, bool value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = register_descriptions_.find(name);
    if (it == register_descriptions_.end()) return;
    if (it->second.mode != RegisterStorageMode::WARP_BITMASK) return;
    if (warp_id < 0 || warp_id >= (int)it->second.predicate_mask_storage.size()) return;
    if (value) {
        it->second.predicate_mask_storage[warp_id] |= (1u << lane_id);
    } else {
        it->second.predicate_mask_storage[warp_id] &= ~(1u << lane_id);
    }
}

void RegisterBankManager::preallocate_registers(
    const std::vector<RegisterInfo> &registers) {
    for (const auto &reg_info : registers) {
        if (reg_info.index == -1) {
            create_register(reg_info.name, reg_info.size);
        } else {
            std::string full_name =
                reg_info.name + std::to_string(reg_info.index);
            create_register(full_name, reg_info.size);
        }
    }
}

void RegisterBankManager::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto &reg_pair : register_descriptions_) {
        auto &desc = reg_pair.second;
        if (desc.mode == RegisterStorageMode::WARP_BITMASK) {
            std::fill(desc.predicate_mask_storage.begin(),
                     desc.predicate_mask_storage.end(), 0u);
        } else {
            for (int w = 0; w < max_warps_; w++) {
                for (int l = 0; l < threads_per_warp_; l++) {
                    std::fill(desc.data_storage[w][l].begin(),
                             desc.data_storage[w][l].end(), 0);
                }
            }
        }
    }
}
