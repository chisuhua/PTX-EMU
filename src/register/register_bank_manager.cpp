#include "register/register_bank_manager.h"
#include <iostream>

RegisterBankManager::RegisterBankManager(int max_warps, int threads_per_warp) 
    : max_warps_(max_warps), threads_per_warp_(threads_per_warp), 
      total_threads_(max_warps * threads_per_warp) {
    // 初始化存储结构
    register_descriptions_.clear();
}

bool RegisterBankManager::create_register(const std::string &name, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 检查寄存器是否已存在
    if (register_descriptions_.find(name) != register_descriptions_.end()) {
        return false; // 寄存器已存在
    }
    
    RegisterDesc desc;
    desc.name = name;
    desc.size = size;
    
    // 初始化存储: [warp_id][lane_id][register_data]
    desc.data_storage.resize(max_warps_);
    for (int w = 0; w < max_warps_; w++) {
        desc.data_storage[w].resize(threads_per_warp_);
        for (int l = 0; l < threads_per_warp_; l++) {
            desc.data_storage[w][l].resize(size, 0); // 初始化为0
        }
    }
    
    register_descriptions_[name] = desc;
    return true;
}

void *RegisterBankManager::get_register(const std::string &name, int warp_id, int lane_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (warp_id >= max_warps_ || warp_id < 0) {
        std::cerr << "Invalid warp_id: " << warp_id << std::endl;
        return nullptr;
    }
    
    if (lane_id >= threads_per_warp_ || lane_id < 0) {
        std::cerr << "Invalid lane_id: " << lane_id << std::endl;
        return nullptr;
    }
    
    auto it = register_descriptions_.find(name);
    if (it == register_descriptions_.end()) {
        std::cerr << "Register not found: " << name << std::endl;
        return nullptr;
    }
    
    if (warp_id >= it->second.data_storage.size()) {
        std::cerr << "Warp ID out of range for register: " << name << std::endl;
        return nullptr;
    }
    
    if (lane_id >= it->second.data_storage[warp_id].size()) {
        std::cerr << "Lane ID out of range for register: " << name << std::endl;
        return nullptr;
    }
    
    return it->second.data_storage[warp_id][lane_id].data();
}

void RegisterBankManager::preallocate_registers(const std::vector<RegisterInfo> &registers) {
    for (const auto &reg_info : registers) {
        std::string full_name = reg_info.name + std::to_string(reg_info.index);
        create_register(full_name, reg_info.size);
    }
}

void RegisterBankManager::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 重置所有寄存器数据为0
    for (auto &reg_pair : register_descriptions_) {
        auto &desc = reg_pair.second;
        for (int w = 0; w < max_warps_; w++) {
            for (int l = 0; l < threads_per_warp_; l++) {
                std::fill(desc.data_storage[w][l].begin(), desc.data_storage[w][l].end(), 0);
            }
        }
    }
}