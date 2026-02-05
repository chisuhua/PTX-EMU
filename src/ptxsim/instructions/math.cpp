#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

// 半精度浮点数转换函数实现
inline float f16_to_f32(uint16_t h) {
    // Simple implementation - for actual use, consider a proper conversion
    // This is a placeholder
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h & 0x7C00) >> 10;
    uint32_t mantissa = h & 0x03FF;
    
    if (exponent == 0) {
        // Denormal
        return std::ldexp((float)mantissa / 1024.0f, -14);
    } else if (exponent == 0x1F) {
        // Infinity or NaN
        return (mantissa == 0) ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    } else {
        // Normal
        exponent += 112;
        mantissa <<= 13;
        uint32_t result = sign | (exponent << 23) | mantissa;
        float f;
        std::memcpy(&f, &result, 4);
        return f;
    }
}

inline uint16_t f32_to_f16(float f) {
    // Simple implementation - for actual use, consider a proper conversion
    // This is a placeholder
    uint32_t x;
    std::memcpy(&x, &f, 4);
    
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t exponent = (x >> 23) & 0xFF;
    uint32_t mantissa = x & 0x7FFFFF;
    
    if (exponent == 0xFF) {
        // Infinity or NaN
        return sign | 0x7C00 | (mantissa ? 0x0200 : 0);
    }
    
    // Convert to half precision
    exponent = exponent - 127 + 15;
    if (exponent >= 0x1F) {
        // Overflow
        return sign | 0x7C00;
    }
    if (exponent <= 0) {
        // Underflow
        return sign;
    }
    
    mantissa >>= 13;
    return sign | (exponent << 10) | mantissa;
}

// 通用模板函数，用于处理一元数学操作
template<typename OpFunc>
void process_unary_math(void *dst, void *src, int bytes, bool is_float, OpFunc op) {
    if (is_float) {
        if (bytes == 4) {
            *(float *)dst = op(*(float *)src);
        } else if (bytes == 8) {
            *(double *)dst = op(*(double *)src);
        } else if (bytes == 2) {
            // 处理半精度浮点数
            uint16_t h;
            std::memcpy(&h, src, 2);
            float f = f16_to_f32(h);
            float result = op(f);
            uint16_t h_result = f32_to_f16(result);
            std::memcpy(dst, &h_result, 2);
        }
    } else {
        // 整数数学运算
        switch (bytes) {
        case 1:
            *(uint8_t *)dst = (uint8_t)op(*(uint8_t *)src);
            break;
        case 2:
            *(uint16_t *)dst = (uint16_t)op(*(uint16_t *)src);
            break;
        case 4:
            *(uint32_t *)dst = (uint32_t)op(*(uint32_t *)src);
            break;
        case 8:
            *(uint64_t *)dst = (uint64_t)op(*(uint64_t *)src);
            break;
        default:
            assert(0 && "Unsupported data size for math operation");
        }
    }
}

void sqrt_operation(ThreadContext *context, void *op[2],
                    const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    process_unary_math(dst, src, bytes, is_float, [](auto x) { return std::sqrt(x); });
}

void sin_operation(ThreadContext *context, void *op[2],
                   const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    process_unary_math(dst, src, bytes, is_float, [](auto x) { return std::sin(x); });
}

void cos_operation(ThreadContext *context, void *op[2],
                   const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    process_unary_math(dst, src, bytes, is_float, [](auto x) { return std::cos(x); });
}

void rcp_operation(ThreadContext *context, void *op[2],
                   const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // RCP指令只支持浮点类型
    assert(is_float && "RCP instruction only supports floating point types");

    process_unary_math(dst, src, bytes, is_float, [](auto x) { return 1.0 / x; });
}

void lg2_operation(ThreadContext *context, void *op[2],
                   const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    assert(is_float && "LG2 instruction only supports floating point types");
    
    process_unary_math(dst, src, bytes, is_float, [](auto x) { return std::log2(x); });
}

void ex2_operation(ThreadContext *context, void *op[2],
                   const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    assert(is_float && "EX2 instruction only supports floating point types");
    
    process_unary_math(dst, src, bytes, is_float, [](auto x) { return std::exp2(x); });
}

void rsqrt_operation(ThreadContext *context, void *op[2],
                     const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    assert(is_float && "RSQRT instruction only supports floating point types");
    
    process_unary_math(dst, src, bytes, is_float, [](auto x) { return 1.0 / std::sqrt(x); });
}
