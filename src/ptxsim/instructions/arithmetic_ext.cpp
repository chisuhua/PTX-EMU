#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include "ptxsim/utils/half_utils.h"
#include "ptxsim/utils/arithmetic_utils.h"
#include <cassert>
#include <cstdint>
#include <cstring>

void ADDC::process_operation(ThreadContext *context, void *op[3],
                             const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];

    // ADDC指令实现带进位的加法：dst = src1 + src2 + carry
    // 从条件码寄存器获取进位值
    uint8_t carry = context->cc_reg.carry ? 1 : 0;

    // 检查是否存在.cc修饰符，决定是否更新条件码寄存器
    bool update_cc = hasCCQualifier(qualifiers);

    if (is_signed) {
        // 有符号整数加法（带进位）
        switch (bytes) {
        case 1: {
            int8_t a, b, r;
            std::memcpy(&a, src1, 1);
            std::memcpy(&b, src2, 1);

            // 计算带进位的加法
            r = a + b + carry;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = (r < 0);
                // 对于有符号数，溢出发生在两个相同符号数相加得到相反符号结果时
                context->cc_reg.overflow = ((a > 0 && b > 0 && r < 0) || (a < 0 && b < 0 && r > 0));
                // 进位标志在结果大于有符号整数最大值或小于最小值时设置
                context->cc_reg.carry = ((a > 0 && b > 0 && r < 0) || (a < 0 && b < 0 && r > 0) || 
                                        (a == 0 && b > 0 && r < 0) || (a > 0 && b == 0 && r < 0));
            }

            std::memcpy(dst, &r, 1);
            break;
        }
        case 2: {
            int16_t a, b, r;
            std::memcpy(&a, src1, 2);
            std::memcpy(&b, src2, 2);

            r = a + b + carry;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = (r < 0);
                context->cc_reg.overflow = ((a > 0 && b > 0 && r < 0) || (a < 0 && b < 0 && r > 0));
                context->cc_reg.carry = ((a > 0 && b > 0 && r < 0) || (a < 0 && b < 0 && r > 0) || 
                                        (a == 0 && b > 0 && r < 0) || (a > 0 && b == 0 && r < 0));
            }

            std::memcpy(dst, &r, 2);
            break;
        }
        case 4: {
            int32_t a, b, r;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);

            r = a + b + carry;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = (r < 0);
                context->cc_reg.overflow = ((a > 0 && b > 0 && r < 0) || (a < 0 && b < 0 && r > 0));
                context->cc_reg.carry = ((a > 0 && b > 0 && r < 0) || (a < 0 && b < 0 && r > 0) || 
                                        (a == 0 && b > 0 && r < 0) || (a > 0 && b == 0 && r < 0));
            }

            std::memcpy(dst, &r, 4);
            break;
        }
        case 8: {
            int64_t a, b, r;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);

            r = a + b + carry;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = (r < 0);
                context->cc_reg.overflow = ((a > 0 && b > 0 && r < 0) || (a < 0 && b < 0 && r > 0));
                context->cc_reg.carry = ((a > 0 && b > 0 && r < 0) || (a < 0 && b < 0 && r > 0) || 
                                        (a == 0 && b > 0 && r < 0) || (a > 0 && b == 0 && r < 0));
            }

            std::memcpy(dst, &r, 8);
            break;
        }
        default:
            assert(0 && "Unsupported data size for signed integer in ADDC");
        }
    } else {
        // 无符号整数加法（带进位）
        switch (bytes) {
        case 1: {
            uint8_t a, b, r;
            std::memcpy(&a, src1, 1);
            std::memcpy(&b, src2, 1);

            // 使用64位临时变量检测溢出
            uint16_t temp = (uint16_t)a + (uint16_t)b + carry;
            r = (uint8_t)temp;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = false;  // 无符号数没有符号标志
                context->cc_reg.overflow = false;  // 无符号数没有溢出标志
                context->cc_reg.carry = (temp > UINT8_MAX);
            }

            std::memcpy(dst, &r, 1);
            break;
        }
        case 2: {
            uint16_t a, b, r;
            std::memcpy(&a, src1, 2);
            std::memcpy(&b, src2, 2);

            uint32_t temp = (uint32_t)a + (uint32_t)b + carry;
            r = (uint16_t)temp;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = false;
                context->cc_reg.overflow = false;
                context->cc_reg.carry = (temp > UINT16_MAX);
            }

            std::memcpy(dst, &r, 2);
            break;
        }
        case 4: {
            uint32_t a, b, r;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);

            // 对于32位无符号整数，需要特殊处理进位
            uint64_t temp = (uint64_t)a + (uint64_t)b + carry;
            r = (uint32_t)temp;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = false;
                context->cc_reg.overflow = false;
                context->cc_reg.carry = (temp > UINT32_MAX);
            }

            std::memcpy(dst, &r, 4);
            break;
        }
        case 8: {
            uint64_t a, b, r;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);

            // 对于64位无符号整数，需要特殊处理进位
            __uint128_t temp = (__uint128_t)a + (__uint128_t)b + carry;
            r = (uint64_t)temp;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = false;
                context->cc_reg.overflow = false;
                context->cc_reg.carry = (temp > UINT64_MAX);
            }

            std::memcpy(dst, &r, 8);
            break;
        }
        default:
            assert(0 && "Unsupported data size for unsigned integer in ADDC");
        }
    }
}

void SUBC::process_operation(ThreadContext *context, void *op[3],
                             const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];

    // SUBC指令实现带借位的减法：dst = src1 - src2 - borrow
    // 从条件码寄存器获取借位值
    uint8_t borrow = context->cc_reg.carry ? 1 : 0;  // SUBC使用carry标志作为borrow

    // 检查是否存在.cc修饰符，决定是否更新条件码寄存器
    bool update_cc = hasCCQualifier(qualifiers);

    if (is_signed) {
        // 有符号整数减法（带借位）
        switch (bytes) {
        case 1: {
            int8_t a, b, r;
            std::memcpy(&a, src1, 1);
            std::memcpy(&b, src2, 1);

            r = a - b - borrow;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = (r < 0);
                // 溢出发生在正数减负数得负数，或负数减正数得正数的情况
                context->cc_reg.overflow = ((a > 0 && b < 0 && r < 0) || (a < 0 && b > 0 && r > 0));
                // 借位标志在a < b + borrow时设置
                context->cc_reg.carry = (a < b + borrow);
            }

            std::memcpy(dst, &r, 1);
            break;
        }
        case 2: {
            int16_t a, b, r;
            std::memcpy(&a, src1, 2);
            std::memcpy(&b, src2, 2);

            r = a - b - borrow;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = (r < 0);
                context->cc_reg.overflow = ((a > 0 && b < 0 && r < 0) || (a < 0 && b > 0 && r > 0));
                context->cc_reg.carry = (a < b + borrow);
            }

            std::memcpy(dst, &r, 2);
            break;
        }
        case 4: {
            int32_t a, b, r;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);

            r = a - b - borrow;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = (r < 0);
                context->cc_reg.overflow = ((a > 0 && b < 0 && r < 0) || (a < 0 && b > 0 && r > 0));
                context->cc_reg.carry = (a < b + borrow);
            }

            std::memcpy(dst, &r, 4);
            break;
        }
        case 8: {
            int64_t a, b, r;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);

            r = a - b - borrow;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = (r < 0);
                context->cc_reg.overflow = ((a > 0 && b < 0 && r < 0) || (a < 0 && b > 0 && r > 0));
                context->cc_reg.carry = (a < b + borrow);
            }

            std::memcpy(dst, &r, 8);
            break;
        }
        default:
            assert(0 && "Unsupported data size for signed integer in SUBC");
        }
    } else {
        // 无符号整数减法（带借位）
        switch (bytes) {
        case 1: {
            uint8_t a, b, r;
            std::memcpy(&a, src1, 1);
            std::memcpy(&b, src2, 1);

            // 使用更大的类型来检测借位
            int16_t temp = (int16_t)a - (int16_t)b - borrow;
            r = (uint8_t)temp;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = false;
                context->cc_reg.overflow = false;
                context->cc_reg.carry = (temp < 0);  // 如果结果为负，说明发生了借位
            }

            std::memcpy(dst, &r, 1);
            break;
        }
        case 2: {
            uint16_t a, b, r;
            std::memcpy(&a, src1, 2);
            std::memcpy(&b, src2, 2);

            int32_t temp = (int32_t)a - (int32_t)b - borrow;
            r = (uint16_t)temp;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = false;
                context->cc_reg.overflow = false;
                context->cc_reg.carry = (temp < 0);
            }

            std::memcpy(dst, &r, 2);
            break;
        }
        case 4: {
            uint32_t a, b, r;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);

            int64_t temp = (int64_t)a - (int64_t)b - borrow;
            r = (uint32_t)temp;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = false;
                context->cc_reg.overflow = false;
                context->cc_reg.carry = (temp < 0);
            }

            std::memcpy(dst, &r, 4);
            break;
        }
        case 8: {
            uint64_t a, b, r;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);

            // 使用128位类型来检测借位
            __int128_t temp = (__int128_t)a - (__int128_t)b - borrow;
            r = (uint64_t)temp;
            
            // 如果有.cc修饰符，则更新条件码寄存器
            if (update_cc) {
                context->cc_reg.zero = (r == 0);
                context->cc_reg.sign = false;
                context->cc_reg.overflow = false;
                context->cc_reg.carry = (temp < 0);
            }

            std::memcpy(dst, &r, 8);
            break;
        }
        default:
            assert(0 && "Unsupported data size for unsigned integer in SUBC");
        }
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