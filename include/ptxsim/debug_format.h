#ifndef DEBUG_FORMAT_H
#define DEBUG_FORMAT_H

#include "utils/logger.h"
#include <any>
#include <sstream>
#include <iomanip>

namespace ptxsim {
namespace debug_format {

// 格式化32位整数（多种进制）
inline std::string format_i32(int32_t value, bool hex = false) {
    if (hex) {
        std::stringstream ss;
        ss << "0x" << std::hex << std::setw(8) << std::setfill('0') << value;
        return ss.str();
    }
    return detail::printf_format("%d", value);
}

// 格式化32位无符号整数
inline std::string format_u32(uint32_t value, bool hex = false) {
    if (hex) {
        std::stringstream ss;
        ss << "0x" << std::hex << std::setw(8) << std::setfill('0') << value;
        return ss.str();
    }
    return detail::printf_format("%u", value);
}

// 格式化64位整数
inline std::string format_i64(int64_t value, bool hex = false) {
    if (hex) {
        std::stringstream ss;
        ss << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
        return ss.str();
    }
    return detail::printf_format("%lld", value);
}

inline std::string format_u64(uint64_t value, bool hex = false) {
    if (hex) {
        std::stringstream ss;
        ss << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
        return ss.str();
    }
    return detail::printf_format("%lld", value);
}

// 格式化32位浮点数
inline std::string format_f32(float value) {
    return detail::printf_format("%f", value);
}

// 格式化64位浮点数
inline std::string format_f64(double value) {
    return detail::printf_format("%f", value);
}

// 格式化谓词值
inline std::string format_pred(bool value) { return value ? "true" : "false"; }

// 格式化内存地址
inline std::string format_address(uint64_t addr) {
    std::stringstream ss;
    ss << "0x" << std::hex << addr;
    return ss.str();
}

// 格式化寄存器值
template <typename T>
inline std::string format_register_value(T value, bool hex = false) {
    if constexpr (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else if constexpr (std::is_floating_point_v<T>) {
        return detail::printf_format("%.6f", value);
    } else if constexpr (std::is_integral_v<T> && sizeof(T) <= 4) {
        if (hex) {
            std::stringstream ss;
            ss << "0x" << std::hex << value;
            return ss.str();
        } else {
            return detail::printf_format("%d", static_cast<int>(value));
        }
    } else if constexpr (std::is_integral_v<T> && sizeof(T) > 4) {
        if (hex) {
            std::stringstream ss;
            ss << "0x" << std::hex << value;
            return ss.str();
        } else {
            return detail::printf_format("%lld", static_cast<long long>(value));
        }
    } else {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }
}

// 专门处理std::any类型的寄存器值格式化
inline std::string format_register_value(const std::any &value,
                                         bool hex = false) {
    if (!value.has_value()) {
        return "null";
    }

    try {
        // 尝试各种常见类型
        if (value.type() == typeid(bool)) {
            return format_register_value(std::any_cast<bool>(value), hex);
        } else if (value.type() == typeid(int8_t)) {
            return format_register_value(std::any_cast<int8_t>(value), hex);
        } else if (value.type() == typeid(uint8_t)) {
            return format_register_value(std::any_cast<uint8_t>(value), hex);
        } else if (value.type() == typeid(int16_t)) {
            return format_register_value(std::any_cast<int16_t>(value), hex);
        } else if (value.type() == typeid(uint16_t)) {
            return format_register_value(std::any_cast<uint16_t>(value), hex);
        } else if (value.type() == typeid(int32_t)) {
            return format_register_value(std::any_cast<int32_t>(value), hex);
        } else if (value.type() == typeid(uint32_t)) {
            return format_register_value(std::any_cast<uint32_t>(value), hex);
        } else if (value.type() == typeid(int64_t)) {
            return format_register_value(std::any_cast<int64_t>(value), hex);
        } else if (value.type() == typeid(uint64_t)) {
            return format_register_value(std::any_cast<uint64_t>(value), hex);
        } else if (value.type() == typeid(float)) {
            return format_register_value(std::any_cast<float>(value), hex);
        } else if (value.type() == typeid(double)) {
            return format_register_value(std::any_cast<double>(value), hex);
        } else {
            return "unknown_type";
        }
    } catch (...) {
        return "format_error";
    }
}

} // namespace debug_format
} // namespace ptxsim

#endif // DEBUG_FORMAT_H