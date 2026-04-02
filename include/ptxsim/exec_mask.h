#ifndef EXEC_MASK_H
#define EXEC_MASK_H

#include <cstdint>
#include <array>
#include <string>

namespace ptxsim {

/**
 * @file exec_mask.h
 * @brief 执行掩码工具类，用于 SIMT 执行的活跃线程管理
 * @details 提供高效的位操作和转换函数
 * @author PTX-EMU Team
 * @date 2026-04-02
 */

class ExecMask {
public:
    // 默认构造函数 (全活跃)
    constexpr ExecMask() : mask_(0xFFFFFFFF) {}
    
    // 从 uint32_t 构造
    explicit constexpr ExecMask(uint32_t mask) : mask_(mask) {}
    
    // 从 lane 活跃状态构造
    ExecMask(const std::array<bool, 32>& active_lanes) {
        mask_ = 0;
        for (int i = 0; i < 32; ++i) {
            if (active_lanes[i]) {
                mask_ |= (1u << i);
            }
        }
    }
    
    // 设置特定 lane 为活跃
    constexpr void set_lane(int lane_id, bool active) {
        if (lane_id >= 0 && lane_id < 32) {
            if (active) {
                mask_ |= (1u << lane_id);
            } else {
                mask_ &= ~(1u << lane_id);
            }
        }
    }
    
    // 检查特定 lane 是否活跃
    constexpr bool is_lane_active(int lane_id) const {
        if (lane_id >= 0 && lane_id < 32) {
            return (mask_ & (1u << lane_id)) != 0;
        }
        return false;
    }
    
    // 获取活跃 lane 数量
    int count_active_lanes() const {
        return __builtin_popcount(mask_);
    }
    
    // 获取下一个活跃 lane (用于迭代)
    int next_active_lane(int current_lane) const {
        for (int i = current_lane + 1; i < 32; ++i) {
            if (is_lane_active(i)) {
                return i;
            }
        }
        return -1;  // 无更多活跃 lane
    }
    
    // 获取第一个活跃 lane
    int first_active_lane() const {
        if (mask_ == 0) return -1;
        return __builtin_ctz(mask_);  // Count trailing zeros
    }
    
    // 检查是否为空 (无活跃线程)
    constexpr bool is_empty() const {
        return mask_ == 0;
    }
    
    // 检查是否全活跃
    constexpr bool is_full() const {
        return mask_ == 0xFFFFFFFF;
    }
    
    // 获取底层掩码值
    constexpr uint32_t value() const {
        return mask_;
    }
    
    // 重置为全活跃
    constexpr void reset() {
        mask_ = 0xFFFFFFFF;
    }
    
    // 清空所有活跃
    constexpr void clear() {
        mask_ = 0;
    }
    
    // 与操作 (用于条件执行)
    constexpr ExecMask operator&(const ExecMask& other) const {
        return ExecMask(mask_ & other.mask_);
    }
    
    // 或操作 (用于合并)
    constexpr ExecMask operator|(const ExecMask& other) const {
        return ExecMask(mask_ | other.mask_);
    }
    
    // 异或操作
    constexpr ExecMask operator^(const ExecMask& other) const {
        return ExecMask(mask_ ^ other.mask_);
    }
    
    // 取反
    constexpr ExecMask operator~() const {
        return ExecMask(~mask_);
    }
    
    // 赋值操作
    constexpr ExecMask& operator=(uint32_t value) {
        mask_ = value;
        return *this;
    }
    
    // 相等比较
    constexpr bool operator==(const ExecMask& other) const {
        return mask_ == other.mask_;
    }
    
    constexpr bool operator!=(const ExecMask& other) const {
        return mask_ != other.mask_;
    }
    
    // 转为字符串 (用于调试)
    std::string to_string() const {
        std::string result = "[";
        for (int i = 0; i < 32; ++i) {
            result += is_lane_active(i) ? "1" : "0";
            if (i < 31) result += ",";
        }
        result += "]";
        return result;
    }
    
private:
    uint32_t mask_;
};

// 辅助函数：从 PTX predicate 计算 exec mask
inline ExecMask exec_mask_from_predicate(const bool* predicates, int warp_size = 32) {
    ExecMask mask;
    for (int i = 0; i < warp_size; ++i) {
        mask.set_lane(i, predicates[i]);
    }
    return mask;
}

} // namespace ptxsim

#endif // EXEC_MASK_H
