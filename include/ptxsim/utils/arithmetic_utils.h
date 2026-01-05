#ifndef PTXSIM_ARITHMETIC_UTILS_H
#define PTXSIM_ARITHMETIC_UTILS_H

#include "ptxsim/thread_context.h"
#include "ptxsim/utils/half_utils.h"
#include <cassert>
#include <cmath> // 添加cmath头文件以支持std::fmod
#include <cstring>
#include <type_traits>

// 通用模板函数，用于处理二元算术操作（带条件码更新）
template <typename OpFunc>
void process_binary_arithmetic(ThreadContext *context, void *dst, void *src1,
                               void *src2, int bytes, bool is_float,
                               bool is_signed, bool update_cc, OpFunc op) {
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

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               (r < 0));
                    // 溢出：两个正数相加得负数，或两个负数相加得正数
                    bool overflow = ((a > 0 && b > 0 && r < 0) ||
                                     (a < 0 && b < 0 && r > 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               overflow);
                    // 有符号数的进位标志通常不使用，但这里设置为溢出标志
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               overflow);
                }

                std::memcpy(dst, &r, 1);
                break;
            }
            case 2: {
                int16_t a, b, r;
                std::memcpy(&a, src1, 2);
                std::memcpy(&b, src2, 2);
                r = op(a, b);

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               (r < 0));
                    bool overflow = ((a > 0 && b > 0 && r < 0) ||
                                     (a < 0 && b < 0 && r > 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               overflow);
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               overflow);
                }

                std::memcpy(dst, &r, 2);
                break;
            }
            case 4: {
                int32_t a, b, r;
                std::memcpy(&a, src1, 4);
                std::memcpy(&b, src2, 4);
                r = op(a, b);

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               (r < 0));
                    bool overflow = ((a > 0 && b > 0 && r < 0) ||
                                     (a < 0 && b < 0 && r > 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               overflow);
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               overflow);
                }

                std::memcpy(dst, &r, 4);
                break;
            }
            case 8: {
                int64_t a, b, r;
                std::memcpy(&a, src1, 8);
                std::memcpy(&b, src2, 8);
                r = op(a, b);

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               (r < 0));
                    bool overflow = ((a > 0 && b > 0 && r < 0) ||
                                     (a < 0 && b < 0 && r > 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               overflow);
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               overflow);
                }

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
                // 使用更大的类型检测溢出
                uint16_t temp_result = op((uint16_t)a, (uint16_t)b);
                r = (uint8_t)temp_result;

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               false); // 无符号数没有符号标志
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               false); // 无符号数没有溢出标志
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               (temp_result > UINT8_MAX));
                }

                std::memcpy(dst, &r, 1);
                break;
            }
            case 2: {
                uint16_t a, b, r;
                std::memcpy(&a, src1, 2);
                std::memcpy(&b, src2, 2);
                uint32_t temp_result = op((uint32_t)a, (uint32_t)b);
                r = (uint16_t)temp_result;

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               (temp_result > UINT16_MAX));
                }

                std::memcpy(dst, &r, 2);
                break;
            }
            case 4: {
                uint32_t a, b, r;
                std::memcpy(&a, src1, 4);
                std::memcpy(&b, src2, 4);
                uint64_t temp_result = op((uint64_t)a, (uint64_t)b);
                r = (uint32_t)temp_result;

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               (temp_result > UINT32_MAX));
                }

                std::memcpy(dst, &r, 4);
                break;
            }
            case 8: {
                uint64_t a, b, r;
                std::memcpy(&a, src1, 8);
                std::memcpy(&b, src2, 8);
                // 使用128位类型检测溢出
                __uint128_t temp_result = op((__uint128_t)a, (__uint128_t)b);
                r = (uint64_t)temp_result;

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               (temp_result > UINT64_MAX));
                }

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
template <typename OpFunc>
void process_unary_arithmetic(ThreadContext *context, void *dst, void *src,
                              int bytes, bool is_float, bool is_signed,
                              bool update_cc, OpFunc op) {
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

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               (r < 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               false); // 一元运算通常不涉及溢出
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               false); // 一元运算通常不设置进位
                }

                std::memcpy(dst, &r, 1);
                break;
            }
            case 2: {
                int16_t a, r;
                std::memcpy(&a, src, 2);
                r = op(a);

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               (r < 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               false);
                }

                std::memcpy(dst, &r, 2);
                break;
            }
            case 4: {
                int32_t a, r;
                std::memcpy(&a, src, 4);
                r = op(a);

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               (r < 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               false);
                }

                std::memcpy(dst, &r, 4);
                break;
            }
            case 8: {
                int64_t a, r;
                std::memcpy(&a, src, 8);
                r = op(a);

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               (r < 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               false);
                }

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

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               false); // 无符号数没有符号标志
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               false);
                }

                std::memcpy(dst, &r, 1);
                break;
            }
            case 2: {
                uint16_t a, r;
                std::memcpy(&a, src, 2);
                r = op(a);

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               false);
                }

                std::memcpy(dst, &r, 2);
                break;
            }
            case 4: {
                uint32_t a, r;
                std::memcpy(&a, src, 4);
                r = op(a);

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX,
                                               (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX,
                                               false);
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX,
                                               false);
                }

                std::memcpy(dst, &r, 4);
                break;
            }
            case 8: {
                uint64_t a, r;
                std::memcpy(&a, src, 8);
                r = op(a);

                // 如果有.cc修饰符，则更新条件码寄存器
                if (update_cc) {
                    context->cc_reg.set_cc_reg(context->cc_reg.ZERO_INDEX, (r == 0));
                    context->cc_reg.set_cc_reg(context->cc_reg.SIGN_INDEX, false);
                    context->cc_reg.set_cc_reg(context->cc_reg.OVERFLOW_INDEX, false);
                    context->cc_reg.set_cc_reg(context->cc_reg.CARRY_INDEX, false);
                }

                std::memcpy(dst, &r, 8);
                break;
            }
            default:
                assert(0 && "Unsupported data size for unsigned integer");
            }
        }
    }
}

#endif // PTXSIM_ARITHMETIC_UTILS_H