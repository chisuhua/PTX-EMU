#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>
#include <type_traits>

// 半精度浮点数转换函数声明
inline float f16_to_f32(uint16_t h);
inline uint16_t f32_to_f16(float f);

// 通用模板函数，用于处理二元算术操作
template<typename OpFunc>
void process_binary_arithmetic(void *dst, void *src1, void *src2, int bytes, 
                              bool is_float, bool is_signed, OpFunc op) {
    if (is_float) {
        // 浮点运算
        switch (bytes) {
        case 2: {
            // 需要 f16 支持（简化：转 f32 计算）
            uint16_t h1, h2;
            std::memcpy(&h1, src1, 2);
            std::memcpy(&h2, src2, 2);

            float f1 = f16_to_f32(h1);
            float f2 = f16_to_f32(h2);
            float result = op(f1, f2);
            uint16_t h_result = f32_to_f16(result);

            std::memcpy(dst, &h_result, 2);
            break;
        }
        case 4: {
            float a, b, r;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);
            r = op(a, b);
            std::memcpy(dst, &r, 4);
            break;
        }
        case 8: {
            double a, b, r;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);
            r = op(a, b);
            std::memcpy(dst, &r, 8);
            break;
        }
        default:
            assert(0 && "Unsupported data size for floating point");
        }
    } else {
        // 整数运算
        if (is_signed) {
            // 有符号整数
            switch (bytes) {
            case 1: {
                int8_t a, b, r;
                std::memcpy(&a, src1, 1);
                std::memcpy(&b, src2, 1);
                r = op(a, b);
                std::memcpy(dst, &r, 1);
                break;
            }
            case 2: {
                int16_t a, b, r;
                std::memcpy(&a, src1, 2);
                std::memcpy(&b, src2, 2);
                r = op(a, b);
                std::memcpy(dst, &r, 2);
                break;
            }
            case 4: {
                int32_t a, b, r;
                std::memcpy(&a, src1, 4);
                std::memcpy(&b, src2, 4);
                r = op(a, b);
                std::memcpy(dst, &r, 4);
                break;
            }
            case 8: {
                int64_t a, b, r;
                std::memcpy(&a, src1, 8);
                std::memcpy(&b, src2, 8);
                r = op(a, b);
                std::memcpy(dst, &r, 8);
                break;
            }
            default:
                assert(0 && "Unsupported data size for signed integer");
            }
        } else {
            // 无符号整数
            switch (bytes) {
            case 1: {
                uint8_t a, b, r;
                std::memcpy(&a, src1, 1);
                std::memcpy(&b, src2, 1);
                r = op(a, b);
                std::memcpy(dst, &r, 1);
                break;
            }
            case 2: {
                uint16_t a, b, r;
                std::memcpy(&a, src1, 2);
                std::memcpy(&b, src2, 2);
                r = op(a, b);
                std::memcpy(dst, &r, 2);
                break;
            }
            case 4: {
                uint32_t a, b, r;
                std::memcpy(&a, src1, 4);
                std::memcpy(&b, src2, 4);
                r = op(a, b);
                std::memcpy(dst, &r, 4);
                break;
            }
            case 8: {
                uint64_t a, b, r;
                std::memcpy(&a, src1, 8);
                std::memcpy(&b, src2, 8);
                r = op(a, b);
                std::memcpy(dst, &r, 8);
                break;
            }
            default:
                assert(0 && "Unsupported data size for unsigned integer");
            }
        }
    }
}

// 通用模板函数，用于处理一元算术操作
template<typename OpFunc>
void process_unary_arithmetic(void *dst, void *src, int bytes, 
                             bool is_float, bool is_signed, OpFunc op) {
    if (is_float) {
        // 浮点运算
        switch (bytes) {
        case 2: {
            // 需要 f16 支持（简化：转 f32 计算）
            uint16_t h;
            std::memcpy(&h, src, 2);

            float f = f16_to_f32(h);
            float result = op(f);
            uint16_t h_result = f32_to_f16(result);

            std::memcpy(dst, &h_result, 2);
            break;
        }
        case 4: {
            float a, r;
            std::memcpy(&a, src, 4);
            r = op(a);
            std::memcpy(dst, &r, 4);
            break;
        }
        case 8: {
            double a, r;
            std::memcpy(&a, src, 8);
            r = op(a);
            std::memcpy(dst, &r, 8);
            break;
        }
        default:
            assert(0 && "Unsupported data size for floating point");
        }
    } else {
        // 整数运算
        if (is_signed) {
            // 有符号整数
            switch (bytes) {
            case 1: {
                int8_t a, r;
                std::memcpy(&a, src, 1);
                r = op(a);
                std::memcpy(dst, &r, 1);
                break;
            }
            case 2: {
                int16_t a, r;
                std::memcpy(&a, src, 2);
                r = op(a);
                std::memcpy(dst, &r, 2);
                break;
            }
            case 4: {
                int32_t a, r;
                std::memcpy(&a, src, 4);
                r = op(a);
                std::memcpy(dst, &r, 4);
                break;
            }
            case 8: {
                int64_t a, r;
                std::memcpy(&a, src, 8);
                r = op(a);
                std::memcpy(dst, &r, 8);
                break;
            }
            default:
                assert(0 && "Unsupported data size for signed integer");
            }
        } else {
            // 无符号整数
            switch (bytes) {
            case 1: {
                uint8_t a, r;
                std::memcpy(&a, src, 1);
                r = op(a);
                std::memcpy(dst, &r, 1);
                break;
            }
            case 2: {
                uint16_t a, r;
                std::memcpy(&a, src, 2);
                r = op(a);
                std::memcpy(dst, &r, 2);
                break;
            }
            case 4: {
                uint32_t a, r;
                std::memcpy(&a, src, 4);
                r = op(a);
                std::memcpy(dst, &r, 4);
                break;
            }
            case 8: {
                uint64_t a, r;
                std::memcpy(&a, src, 8);
                r = op(a);
                std::memcpy(dst, &r, 8);
                break;
            }
            default:
                assert(0 && "Unsupported data size for unsigned integer");
            }
        }
    }
}

void ADD::process_operation(ThreadContext *context, void *op[3],
                            const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];

    // 使用模板函数执行加法操作
    process_binary_arithmetic(dst, src1, src2, bytes, is_float, is_signed,
        [](auto a, auto b) { return a + b; });
}

void SUB::process_operation(ThreadContext *context, void *op[3],
                            const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];

    // 使用模板函数执行减法操作
    process_binary_arithmetic(dst, src1, src2, bytes, is_float, is_signed,
        [](auto a, auto b) { return a - b; });
}

// Helper function to convert f16 to f32 (simplified)
inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 1;
    uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f32;

    if (exp == 0x1F) {
        // Infinity or NaN
        f32 = (sign << 31) | (0xFFU << 23) | (mant << 13);
    } else {
        f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }

    float res;
    std::memcpy(&res, &f32, 4);
    return res;
}

// Helper function to convert f32 to f16 (simplified)
inline uint16_t f32_to_f16(float f) {
    uint32_t f32;
    std::memcpy(&f32, &f, 4);

    uint32_t sign = (f32 >> 31) & 1;
    uint32_t exp = (f32 >> 23) & 0xFF;
    uint32_t mant = f32 & 0x7FFFFF;

    uint16_t h;
    if (exp == 0xFF) {
        // Infinity or NaN
        h = (sign << 15) | (0x1F << 10) | (mant ? 0x200 : 0) | (mant >> 13);
    } else {
        int32_t new_exp = static_cast<int32_t>(exp) - 112;
        if (new_exp >= 31) {
            // Overflow
            h = (sign << 15) | (0x1F << 10); // Infinity
        } else if (new_exp <= 0) {
            // Underflow
            h = (sign << 15); // Zero or subnormal
        } else {
            h = (sign << 15) | (new_exp << 10) | (mant >> 13);
        }
    }

    return h;
}

void MUL::process_operation(ThreadContext *context, void *op[3],
                            const std::vector<Qualifier> &qualifiers) {
    // === 解析类型和修饰符 ===
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];

    // 检查修饰符
    bool has_wide = QvecHasQ(qualifiers, Qualifier::Q_WIDE);
    bool has_hi = QvecHasQ(qualifiers, Qualifier::Q_HI);
    bool has_lo = QvecHasQ(qualifiers, Qualifier::Q_LO);

    // === 浮点类型：直接相乘（忽略修饰符）===
    if (is_float) {
        if (bytes == 2) { // Q_F16
            // 需要 f16 支持（简化：转 f32 计算）
            uint16_t h1, h2;
            std::memcpy(&h1, src1, 2);
            std::memcpy(&h2, src2, 2);
            float f1 = f16_to_f32(h1);
            float f2 = f16_to_f32(h2);
            float res = f1 * f2;
            uint16_t h_res = f32_to_f16(res);
            std::memcpy(dst, &h_res, 2);
        } else if (bytes == 4) { // Q_F32
            float a, b, r;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);
            r = a * b;
            std::memcpy(dst, &r, 4);
        } else if (bytes == 8) { // Q_F64
            double a, b, r;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);
            r = a * b;
            std::memcpy(dst, &r, 8);
        }
        return;
    }

    // === 整数类型：处理 wide/hi/lo ===
    // 读取操作数为 uint64_t/int64_t（足够容纳乘积）
    uint64_t u1 = 0, u2 = 0;
    int64_t s1 = 0, s2 = 0;

    switch (bytes) {
    case 1: {
        if (is_signed) {
            int8_t a, b;
            std::memcpy(&a, src1, 1);
            std::memcpy(&b, src2, 1);
            s1 = a;
            s2 = b;
        } else {
            uint8_t a, b;
            std::memcpy(&a, src1, 1);
            std::memcpy(&b, src2, 1);
            u1 = a;
            u2 = b;
        }
        break;
    }
    case 2: {
        if (is_signed) {
            int16_t a, b;
            std::memcpy(&a, src1, 2);
            std::memcpy(&b, src2, 2);
            s1 = a;
            s2 = b;
        } else {
            uint16_t a, b;
            std::memcpy(&a, src1, 2);
            std::memcpy(&b, src2, 2);
            u1 = a;
            u2 = b;
        }
        break;
    }
    case 4: {
        if (is_signed) {
            int32_t a, b;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);
            s1 = a;
            s2 = b;
        } else {
            uint32_t a, b;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);
            u1 = a;
            u2 = b;
        }
        break;
    }
    case 8: {
        if (is_signed) {
            int64_t a, b;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);
            s1 = a;
            s2 = b;
        } else {
            uint64_t a, b;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);
            u1 = a;
            u2 = b;
        }
        break;
    }
    }

    // === 执行乘法 ===
    if (has_wide) {
        // wide: 结果宽度 = 2 * src_width
        uint64_t result =
            (is_signed) ? static_cast<uint64_t>(s1 * s2) : (u1 * u2);

        size_t dst_size = 2 * bytes;
        std::memcpy(dst, &result, dst_size);

    } else if (has_hi) {
        // hi: 取高半部分
        uint64_t full;
        if (is_signed) {
            // 有符号高32位：需算术右移
            full = static_cast<uint64_t>(s1 * s2);
        } else {
            full = u1 * u2;
        }

        uint64_t hi = (bytes == 4)   ? (full >> 32)
                      : (bytes == 2) ? (full >> 16)
                      : (bytes == 1) ? (full >> 8)
                                     : 0;

        std::memcpy(dst, &hi, bytes);

    } else { // Q_LO or Q_NONE
        // lo: 取低半部分（普通乘法）
        uint64_t full =
            (is_signed) ? static_cast<uint64_t>(s1 * s2) : (u1 * u2);

        std::memcpy(dst, &full, bytes);
    }
}

void MUL24::process_operation(ThreadContext *context, void *op[3],
                              const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];

    // 检查修饰符
    bool has_hi = QvecHasQ(qualifiers, Qualifier::Q_HI);
    bool has_lo = QvecHasQ(qualifiers, Qualifier::Q_LO);

    // MUL24指令处理32位操作数，但只使用其中的24位（最低有效位）
    // 结果是48位，根据修饰符选择高32位或低32位
    if (bytes == 4) { // 仅支持32位操作数 (.u32 或 .s32)
        uint32_t a, b;
        std::memcpy(&a, src1, 4);
        std::memcpy(&b, src2, 4);

        // 只保留操作数的低24位
        uint32_t a24 = a & 0xFFFFFF;
        uint32_t b24 = b & 0xFFFFFF;

        // 执行24位乘法，得到48位结果
        uint64_t result =
            static_cast<uint64_t>(a24) * static_cast<uint64_t>(b24);

        // 根据修饰符选择结果的哪一部分
        if (has_hi) {
            // 取高32位 (47..16)
            uint32_t hi_part = (result >> 16) & 0xFFFFFFFF;
            std::memcpy(dst, &hi_part, 4);
        } else {
            // 默认或LO模式：取低32位 (31..0)
            uint32_t lo_part = result & 0xFFFFFFFF;
            std::memcpy(dst, &lo_part, 4);
        }
    } else {
        // 不支持的数据大小
        assert(0 && "MUL24 only supports 32-bit operands (.u32 or .s32)");
    }
}

void DIV::process_operation(ThreadContext *context, void *op[3],
                            const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];

    // 使用模板函数执行除法操作
    process_binary_arithmetic(dst, src1, src2, bytes, is_float, is_signed,
        [](auto a, auto b) { return a / b; });
}

void MAD::process_operation(ThreadContext *context, void *op[4],
                            const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];
    void *src3 = op[3];

    // 检查修饰符
    bool has_wide = QvecHasQ(qualifiers, Qualifier::Q_WIDE);
    bool has_hi = QvecHasQ(qualifiers, Qualifier::Q_HI);
    bool has_lo = QvecHasQ(qualifiers, Qualifier::Q_LO);
    bool has_sat = QvecHasQ(qualifiers, Qualifier::Q_SAT);

    // === 浮点类型：直接执行乘加（忽略修饰符）===
    if (is_float) {
        if (bytes == 2) { // Q_F16
            // 需要 f16 支持（简化：转 f32 计算）
            uint16_t h1, h2, h3;
            std::memcpy(&h1, src1, 2);
            std::memcpy(&h2, src2, 2);
            std::memcpy(&h3, src3, 2);
            float f1 = f16_to_f32(h1);
            float f2 = f16_to_f32(h2);
            float f3 = f16_to_f32(h3);
            float res = f1 * f2 + f3;
            uint16_t h_res = f32_to_f16(res);
            std::memcpy(dst, &h_res, 2);
        } else if (bytes == 4) { // Q_F32
            float a, b, c, r;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);
            std::memcpy(&c, src3, 4);
            r = a * b + c;
            std::memcpy(dst, &r, 4);
        } else if (bytes == 8) { // Q_F64
            double a, b, c, r;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);
            std::memcpy(&c, src3, 8);
            r = a * b + c;
            std::memcpy(dst, &r, 8);
        }
        return;
    }

    // === 整数类型：处理 wide/hi/lo/sat ===
    // 读取操作数为 uint64_t/int64_t（足够容纳乘积）
    uint64_t u1 = 0, u2 = 0, u3 = 0;
    int64_t s1 = 0, s2 = 0, s3 = 0;

    switch (bytes) {
    case 1: {
        if (is_signed) {
            int8_t a, b, c;
            std::memcpy(&a, src1, 1);
            std::memcpy(&b, src2, 1);
            std::memcpy(&c, src3, 1);
            s1 = a;
            s2 = b;
            s3 = c;
        } else {
            uint8_t a, b, c;
            std::memcpy(&a, src1, 1);
            std::memcpy(&b, src2, 1);
            std::memcpy(&c, src3, 1);
            u1 = a;
            u2 = b;
            u3 = c;
        }
        break;
    }
    case 2: {
        if (is_signed) {
            int16_t a, b, c;
            std::memcpy(&a, src1, 2);
            std::memcpy(&b, src2, 2);
            std::memcpy(&c, src3, 2);
            s1 = a;
            s2 = b;
            s3 = c;
        } else {
            uint16_t a, b, c;
            std::memcpy(&a, src1, 2);
            std::memcpy(&b, src2, 2);
            std::memcpy(&c, src3, 2);
            u1 = a;
            u2 = b;
            u3 = c;
        }
        break;
    }
    case 4: {
        if (is_signed) {
            int32_t a, b, c;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);
            std::memcpy(&c, src3, 4);
            s1 = a;
            s2 = b;
            s3 = c;
        } else {
            uint32_t a, b, c;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);
            std::memcpy(&c, src3, 4);
            u1 = a;
            u2 = b;
            u3 = c;
        }
        break;
    }
    case 8: {
        if (is_signed) {
            int64_t a, b, c;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);
            std::memcpy(&c, src3, 8);
            s1 = a;
            s2 = b;
            s3 = c;
        } else {
            uint64_t a, b, c;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);
            std::memcpy(&c, src3, 8);
            u1 = a;
            u2 = b;
            u3 = c;
        }
        break;
    }
    }

    // === 执行乘加操作 ===
    if (has_wide) {
        // wide: 结果宽度 = 2 * src_width
        uint64_t mul_result =
            (is_signed) ? static_cast<uint64_t>(s1 * s2) : (u1 * u2);

        uint64_t add_operand = (is_signed) ? static_cast<uint64_t>(s3) : u3;

        // 对于wide模式，我们需要一个更宽的结果类型
        if (bytes == 2) { // 16-bit -> 32-bit
            uint32_t result = static_cast<uint32_t>(mul_result) +
                              static_cast<uint32_t>(add_operand);
            std::memcpy(dst, &result, 4);
        } else if (bytes == 4) { // 32-bit -> 64-bit
            uint64_t result = mul_result + add_operand;
            std::memcpy(dst, &result, 8);
        } else {
            assert(0 &&
                   "WIDE mode only supported for 16-bit and 32-bit integers");
        }

    } else if (has_hi) {
        // hi: 取高半部分
        uint64_t mul_full;
        if (is_signed) {
            mul_full = static_cast<uint64_t>(s1 * s2);
        } else {
            mul_full = u1 * u2;
        }

        uint64_t add_operand = (is_signed) ? static_cast<uint64_t>(s3) : u3;
        uint64_t result = mul_full + add_operand;

        // SAT模式只适用于.s32类型和HI模式
        if (has_sat && bytes == 4 && is_signed) {
            // 限制结果在32位有符号整数范围内
            const int32_t MAX_INT32 = 0x7FFFFFFF;
            const int32_t MIN_INT32 = 0x80000000;

            if (static_cast<int64_t>(result) > MAX_INT32) {
                result = MAX_INT32;
            } else if (static_cast<int64_t>(result) < MIN_INT32) {
                result = MIN_INT32;
            }
        }

        uint64_t hi = (bytes == 4)   ? (result >> 32)
                      : (bytes == 2) ? (result >> 16)
                      : (bytes == 1) ? (result >> 8)
                                     : 0;

        std::memcpy(dst, &hi, bytes);

    } else { // Q_LO or Q_NONE
        // lo: 取低半部分（普通乘加）
        uint64_t mul_full =
            (is_signed) ? static_cast<uint64_t>(s1 * s2) : (u1 * u2);

        uint64_t add_operand = (is_signed) ? static_cast<uint64_t>(s3) : u3;
        uint64_t result = mul_full + add_operand;

        std::memcpy(dst, &result, bytes);
    }
}

void MAD24::process_operation(ThreadContext *context, void *op[4],
                              const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];
    void *src3 = op[3];

    // 检查修饰符
    bool has_hi = QvecHasQ(qualifiers, Qualifier::Q_HI);
    bool has_lo = QvecHasQ(qualifiers, Qualifier::Q_LO);

    // MAD24指令处理32位操作数，但只使用其中的24位（最低有效位）
    // 结果是48位加上第三个操作数，根据修饰符选择高32位或低32位
    if (bytes == 4) { // 仅支持32位操作数 (.u32 或 .s32)
        uint32_t a, b, c;
        std::memcpy(&a, src1, 4);
        std::memcpy(&b, src2, 4);
        std::memcpy(&c, src3, 4);

        // 只保留操作数的低24位
        uint32_t a24 = a & 0xFFFFFF;
        uint32_t b24 = b & 0xFFFFFF;

        // 执行24位乘法，得到48位结果
        uint64_t mul_result =
            static_cast<uint64_t>(a24) * static_cast<uint64_t>(b24);

        // 加上第三个操作数
        uint64_t result = mul_result + static_cast<uint64_t>(c);

        // 根据修饰符选择结果的哪一部分
        if (has_hi) {
            // 取高32位 (63..32)
            uint32_t hi_part = (result >> 32) & 0xFFFFFFFF;
            std::memcpy(dst, &hi_part, 4);
        } else {
            // 默认或LO模式：取低32位 (31..0)
            uint32_t lo_part = result & 0xFFFFFFFF;
            std::memcpy(dst, &lo_part, 4);
        }
    } else {
        // 不支持的数据大小
        assert(0 && "MAD24 only supports 32-bit operands (.u32 or .s32)");
    }
}

void FMA::process_operation(ThreadContext *context, void *op[4],
                            const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];
    void *src3 = op[3];

    // 检查修饰符
    bool has_wide = QvecHasQ(qualifiers, Qualifier::Q_WIDE);
    bool has_hi = QvecHasQ(qualifiers, Qualifier::Q_HI);
    bool has_lo = QvecHasQ(qualifiers, Qualifier::Q_LO);
    bool has_sat = QvecHasQ(qualifiers, Qualifier::Q_SAT);

    // === 浮点类型：执行融合乘加（忽略修饰符，因为FMA本身就是融合操作）===
    if (is_float) {
        if (bytes == 2) { // Q_F16
            // 需要 f16 支持（简化：转 f32 计算）
            uint16_t h1, h2, h3;
            std::memcpy(&h1, src1, 2);
            std::memcpy(&h2, src2, 2);
            std::memcpy(&h3, src3, 2);
            float f1 = f16_to_f32(h1);
            float f2 = f16_to_f32(h2);
            float f3 = f16_to_f32(h3);
            // FMA操作：f1 * f2 + f3，只进行一次舍入
            float res = f1 * f2 + f3;
            uint16_t h_res = f32_to_f16(res);
            std::memcpy(dst, &h_res, 2);
        } else if (bytes == 4) { // Q_F32
            float a, b, c, r;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);
            std::memcpy(&c, src3, 4);
            // FMA操作：a * b + c，只进行一次舍入
            r = a * b + c;
            std::memcpy(dst, &r, 4);
        } else if (bytes == 8) { // Q_F64
            double a, b, c, r;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);
            std::memcpy(&c, src3, 8);
            // FMA操作：a * b + c，只进行一次舍入
            r = a * b + c;
            std::memcpy(dst, &r, 8);
        }
        return;
    }

    // === 整数类型：处理 wide/hi/lo/sat ===
    // 读取操作数为 uint64_t/int64_t（足够容纳乘积）
    uint64_t u1 = 0, u2 = 0, u3 = 0;
    int64_t s1 = 0, s2 = 0, s3 = 0;

    switch (bytes) {
    case 1: {
        if (is_signed) {
            int8_t a, b, c;
            std::memcpy(&a, src1, 1);
            std::memcpy(&b, src2, 1);
            std::memcpy(&c, src3, 1);
            s1 = a;
            s2 = b;
            s3 = c;
        } else {
            uint8_t a, b, c;
            std::memcpy(&a, src1, 1);
            std::memcpy(&b, src2, 1);
            std::memcpy(&c, src3, 1);
            u1 = a;
            u2 = b;
            u3 = c;
        }
        break;
    }
    case 2: {
        if (is_signed) {
            int16_t a, b, c;
            std::memcpy(&a, src1, 2);
            std::memcpy(&b, src2, 2);
            std::memcpy(&c, src3, 2);
            s1 = a;
            s2 = b;
            s3 = c;
        } else {
            uint16_t a, b, c;
            std::memcpy(&a, src1, 2);
            std::memcpy(&b, src2, 2);
            std::memcpy(&c, src3, 2);
            u1 = a;
            u2 = b;
            u3 = c;
        }
        break;
    }
    case 4: {
        if (is_signed) {
            int32_t a, b, c;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);
            std::memcpy(&c, src3, 4);
            s1 = a;
            s2 = b;
            s3 = c;
        } else {
            uint32_t a, b, c;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);
            std::memcpy(&c, src3, 4);
            u1 = a;
            u2 = b;
            u3 = c;
        }
        break;
    }
    case 8: {
        if (is_signed) {
            int64_t a, b, c;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);
            std::memcpy(&c, src3, 8);
            s1 = a;
            s2 = b;
            s3 = c;
        } else {
            uint64_t a, b, c;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);
            std::memcpy(&c, src3, 8);
            u1 = a;
            u2 = b;
            u3 = c;
        }
        break;
    }
    }

    // === 执行融合乘加操作 ===
    if (has_wide) {
        // wide: 结果宽度 = 2 * src_width
        uint64_t mul_result =
            (is_signed) ? static_cast<uint64_t>(s1 * s2) : (u1 * u2);

        uint64_t add_operand = (is_signed) ? static_cast<uint64_t>(s3) : u3;

        // 对于wide模式，我们需要一个更宽的结果类型
        if (bytes == 2) { // 16-bit -> 32-bit
            uint32_t result = static_cast<uint32_t>(mul_result) +
                              static_cast<uint32_t>(add_operand);
            std::memcpy(dst, &result, 4);
        } else if (bytes == 4) { // 32-bit -> 64-bit
            uint64_t result = mul_result + add_operand;
            std::memcpy(dst, &result, 8);
        } else {
            assert(0 &&
                   "WIDE mode only supported for 16-bit and 32-bit integers");
        }

    } else if (has_hi) {
        // hi: 取高半部分
        uint64_t mul_full;
        if (is_signed) {
            mul_full = static_cast<uint64_t>(s1 * s2);
        } else {
            mul_full = u1 * u2;
        }

        uint64_t add_operand = (is_signed) ? static_cast<uint64_t>(s3) : u3;
        uint64_t result = mul_full + add_operand;

        // SAT模式只适用于.s32类型和HI模式
        if (has_sat && bytes == 4 && is_signed) {
            // 限制结果在32位有符号整数范围内
            const int32_t MAX_INT32 = 0x7FFFFFFF;
            const int32_t MIN_INT32 = 0x80000000;

            if (static_cast<int64_t>(result) > MAX_INT32) {
                result = MAX_INT32;
            } else if (static_cast<int64_t>(result) < MIN_INT32) {
                result = MIN_INT32;
            }
        }

        uint64_t hi = (bytes == 4)   ? (result >> 32)
                      : (bytes == 2) ? (result >> 16)
                      : (bytes == 1) ? (result >> 8)
                                     : 0;

        std::memcpy(dst, &hi, bytes);

    } else { // Q_LO or Q_NONE
        // lo: 取低半部分（普通乘加）
        uint64_t mul_full =
            (is_signed) ? static_cast<uint64_t>(s1 * s2) : (u1 * u2);

        uint64_t add_operand = (is_signed) ? static_cast<uint64_t>(s3) : u3;
        uint64_t result = mul_full + add_operand;

        std::memcpy(dst, &result, bytes);
    }
}

void NEG::process_operation(ThreadContext *context, void *op[2],
                            const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src = op[1];

    // 使用模板函数执行取负操作
    process_unary_arithmetic(dst, src, bytes, is_float, is_signed,
        [](auto val) { return -val; });
}

void ABS::process_operation(ThreadContext *context, void *op[2],
                            const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src = op[1];

    // 使用模板函数执行绝对值操作
    process_unary_arithmetic(dst, src, bytes, is_float, is_signed,
        [](auto val) {
            if constexpr (std::is_floating_point_v<std::decay_t<decltype(val)>>) {
                return std::abs(val);
            } else if constexpr (std::is_signed_v<std::decay_t<decltype(val)>>) {
                return val < 0 ? -val : val;
            } else {
                return val; // 无符号类型直接返回
            }
        });
}

void MIN::process_operation(ThreadContext *context, void *op[3],
                            const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];

    // 使用模板函数执行最小值操作
    process_binary_arithmetic(dst, src1, src2, bytes, is_float, is_signed,
        [](auto a, auto b) { 
            if constexpr (std::is_floating_point_v<std::decay_t<decltype(a)>>) {
                return std::min(a, b);
            } else {
                return std::min(a, b);
            }
        });
}

void MAX::process_operation(ThreadContext *context, void *op[3],
                            const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];

    // 使用模板函数执行最大值操作
    process_binary_arithmetic(dst, src1, src2, bytes, is_float, is_signed,
        [](auto a, auto b) { 
            if constexpr (std::is_floating_point_v<std::decay_t<decltype(a)>>) {
                return std::max(a, b);
            } else {
                return std::max(a, b);
            }
        });
}

void REM::process_operation(ThreadContext *context, void *op[3],
                            const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];

    // 使用模板函数执行取模操作
    process_binary_arithmetic(dst, src1, src2, bytes, is_float, is_signed,
        [](auto a, auto b) { 
            if constexpr (std::is_floating_point_v<std::decay_t<decltype(a)>>) {
                return std::fmod(a, b); // 对于浮点数使用fmod
            } else if constexpr (std::is_signed_v<std::decay_t<decltype(a)>>) {
                return a % b; // 有符号整数取模
            } else {
                return static_cast<std::make_unsigned_t<std::decay_t<decltype(a)>>>(a) % 
                       static_cast<std::make_unsigned_t<std::decay_t<decltype(b)>>>(b); // 无符号整数取模
            }
        });
}
