#include "ptxsim/instruction_handlers/arithmetic_handler.h"
#include "ptxsim/instruction_processor_utils.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

void ADD::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::ADD *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->addOp[0], ss->addQualifier);
    void *op1 = context->get_operand_addr(ss->addOp[1], ss->addQualifier);
    void *op2 = context->get_operand_addr(ss->addOp[2], ss->addQualifier);

    // 执行ADD操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->addQualifier,
                        (OperandContext::REG *)ss->addOp[0].operand);
}

void ADD::process_operation(ThreadContext *context, void *dst, void *src1,
                            void *src2, std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // 根据数据类型执行加法操作
    switch (bytes) {
    case 1: {
        if (is_float) {
            *(uint8_t *)dst = (*(uint8_t *)src1) + (*(uint8_t *)src2);
        } else {
            *(uint8_t *)dst = (*(uint8_t *)src1) + (*(uint8_t *)src2);
        }
        break;
    }
    case 2: {
        if (is_float) {
            *(uint16_t *)dst = (*(uint16_t *)src1) + (*(uint16_t *)src2);
        } else {
            *(uint16_t *)dst = (*(uint16_t *)src1) + (*(uint16_t *)src2);
        }
        break;
    }
    case 4: {
        if (is_float) {
            *(float *)dst = (*(float *)src1) + (*(float *)src2);
        } else {
            *(uint32_t *)dst = (*(uint32_t *)src1) + (*(uint32_t *)src2);
        }
        break;
    }
    case 8: {
        if (is_float) {
            *(double *)dst = (*(double *)src1) + (*(double *)src2);
        } else {
            *(uint64_t *)dst = (*(uint64_t *)src1) + (*(uint64_t *)src2);
        }
        break;
    }
    default:
        // 不支持的数据大小
        assert(0 && "Unsupported data size for ADD instruction");
    }
}

void SUB::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::SUB *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->subOp[0], ss->subQualifier);
    void *op1 = context->get_operand_addr(ss->subOp[1], ss->subQualifier);
    void *op2 = context->get_operand_addr(ss->subOp[2], ss->subQualifier);

    // 执行SUB操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->subQualifier,
                        (OperandContext::REG *)ss->subOp[0].operand);
}

void SUB::process_operation(ThreadContext *context, void *dst, void *src1,
                            void *src2, std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // 根据数据类型执行减法操作
    switch (bytes) {
    case 1: {
        if (is_float) {
            *(uint8_t *)dst = (*(uint8_t *)src1) - (*(uint8_t *)src2);
        } else {
            *(uint8_t *)dst = (*(uint8_t *)src1) - (*(uint8_t *)src2);
        }
        break;
    }
    case 2: {
        if (is_float) {
            *(uint16_t *)dst = (*(uint16_t *)src1) - (*(uint16_t *)src2);
        } else {
            *(uint16_t *)dst = (*(uint16_t *)src1) - (*(uint16_t *)src2);
        }
        break;
    }
    case 4: {
        if (is_float) {
            *(float *)dst = (*(float *)src1) - (*(float *)src2);
        } else {
            *(uint32_t *)dst = (*(uint32_t *)src1) - (*(uint32_t *)src2);
        }
        break;
    }
    case 8: {
        if (is_float) {
            *(double *)dst = (*(double *)src1) - (*(double *)src2);
        } else {
            *(uint64_t *)dst = (*(uint64_t *)src1) - (*(uint64_t *)src2);
        }
        break;
    }
    default:
        // 不支持的数据大小
        assert(0 && "Unsupported data size for SUB instruction");
    }
}

void MUL::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::MUL *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->mulOp[0], ss->mulQualifier);
    void *op1 = context->get_operand_addr(ss->mulOp[1], ss->mulQualifier);
    void *op2 = context->get_operand_addr(ss->mulOp[2], ss->mulQualifier);

    // 执行MUL操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->mulQualifier,
                        (OperandContext::REG *)ss->mulOp[0].operand);
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

void MUL::process_operation(ThreadContext *context, void *dst, void *src1,
                            void *src2, std::vector<Qualifier> &qualifiers) {
    // === 解析类型和修饰符 ===
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);

    // 检查修饰符
    bool has_wide = context->QvecHasQ(qualifiers, Qualifier::Q_WIDE);
    bool has_hi = context->QvecHasQ(qualifiers, Qualifier::Q_HI);
    bool has_lo = context->QvecHasQ(qualifiers, Qualifier::Q_LO);

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

void MUL24::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::MUL24 *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->mul24Op[0], ss->mul24Qualifier);
    void *op1 = context->get_operand_addr(ss->mul24Op[1], ss->mul24Qualifier);
    void *op2 = context->get_operand_addr(ss->mul24Op[2], ss->mul24Qualifier);

    // 执行MUL24操作
    PROCESS_OPERATION_3(context, to, op1, op2, ss->mul24Qualifier,
                        (OperandContext::REG *)ss->mul24Op[0].operand);
}

void MUL24::process_operation(ThreadContext *context, void *dst, void *src1,
                              void *src2, std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);

    // 检查修饰符
    bool has_hi = context->QvecHasQ(qualifiers, Qualifier::Q_HI);
    bool has_lo = context->QvecHasQ(qualifiers, Qualifier::Q_LO);

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

void DIV::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::DIV *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->divOp[0], ss->divQualifier);
    void *op1 = context->get_operand_addr(ss->divOp[1], ss->divQualifier);
    void *op2 = context->get_operand_addr(ss->divOp[2], ss->divQualifier);

    // 执行DIV操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->divQualifier,
                        (OperandContext::REG *)ss->divOp[0].operand);
}

void DIV::process_operation(ThreadContext *context, void *dst, void *src1,
                            void *src2, std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // 根据数据类型执行除法操作
    switch (bytes) {
    case 1: {
        if (is_float) {
            *(uint8_t *)dst = (*(uint8_t *)src1) / (*(uint8_t *)src2);
        } else {
            *(uint8_t *)dst = (*(uint8_t *)src1) / (*(uint8_t *)src2);
        }
        break;
    }
    case 2: {
        if (is_float) {
            *(uint16_t *)dst = (*(uint16_t *)src1) / (*(uint16_t *)src2);
        } else {
            *(uint16_t *)dst = (*(uint16_t *)src1) / (*(uint16_t *)src2);
        }
        break;
    }
    case 4: {
        if (is_float) {
            *(float *)dst = (*(float *)src1) / (*(float *)src2);
        } else {
            *(uint32_t *)dst = (*(uint32_t *)src1) / (*(uint32_t *)src2);
        }
        break;
    }
    case 8: {
        if (is_float) {
            *(double *)dst = (*(double *)src1) / (*(double *)src2);
        } else {
            *(uint64_t *)dst = (*(uint64_t *)src1) / (*(uint64_t *)src2);
        }
        break;
    }
    default:
        // 不支持的数据大小
        assert(0 && "Unsupported data size for DIV instruction");
    }
}

void MAD::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::MAD *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->madOp[0], ss->madQualifier);
    void *op1 = context->get_operand_addr(ss->madOp[1], ss->madQualifier);
    void *op2 = context->get_operand_addr(ss->madOp[2], ss->madQualifier);
    void *op3 = context->get_operand_addr(ss->madOp[3], ss->madQualifier);

    // 执行MAD操作
    PROCESS_OPERATION_4(context, to, op1, op2, op3, ss->madQualifier,
                        (OperandContext::REG *)ss->madOp[0].operand);
}

void MAD::process_operation(ThreadContext *context, void *dst, void *src1,
                            void *src2, void *src3,
                            std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);

    // 检查修饰符
    bool has_wide = context->QvecHasQ(qualifiers, Qualifier::Q_WIDE);
    bool has_hi = context->QvecHasQ(qualifiers, Qualifier::Q_HI);
    bool has_lo = context->QvecHasQ(qualifiers, Qualifier::Q_LO);
    bool has_sat = context->QvecHasQ(qualifiers, Qualifier::Q_SAT);

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

void FMA::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::FMA *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->fmaOp[0], ss->fmaQualifier);
    void *op1 = context->get_operand_addr(ss->fmaOp[1], ss->fmaQualifier);
    void *op2 = context->get_operand_addr(ss->fmaOp[2], ss->fmaQualifier);
    void *op3 = context->get_operand_addr(ss->fmaOp[3], ss->fmaQualifier);

    // 执行FMA操作
    PROCESS_OPERATION_4(context, to, op1, op2, op3, ss->fmaQualifier,
                        (OperandContext::REG *)ss->fmaOp[0].operand);
}

void FMA::process_operation(ThreadContext *context, void *dst, void *src1,
                            void *src2, void *src3,
                            std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);

    // 检查修饰符
    bool has_wide = context->QvecHasQ(qualifiers, Qualifier::Q_WIDE);
    bool has_hi = context->QvecHasQ(qualifiers, Qualifier::Q_HI);
    bool has_lo = context->QvecHasQ(qualifiers, Qualifier::Q_LO);
    bool has_sat = context->QvecHasQ(qualifiers, Qualifier::Q_SAT);

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
