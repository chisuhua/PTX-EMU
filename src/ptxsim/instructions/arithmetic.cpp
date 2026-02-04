#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/arithmetic_utils.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>
#include <type_traits>

// // 通用模板函数，用于处理二元算术操作
// template <typename OpFunc>
// void process_binary_arithmetic(ThreadContext *context, void *dst, void *src1,
//                                void *src2, int bytes, bool is_float,
//                                bool is_signed, bool update_cc, OpFunc op) {
//     if (is_float) {
//         // 浮点运算
//         switch (bytes) {
//         case 2: {
//             // 需要 f16 支持（简化：转 f32 计算）
//             uint16_t h1, h2;
//             std::memcpy(&h1, src1, 2);
//             std::memcpy(&h2, src2, 2);

//             float f1 = f16_to_f32(h1);
//             float f2 = f16_to_f32(h2);
//             float result = op(f1, f2);
//             uint16_t h_result = f32_to_f16(result);

//             std::memcpy(dst, &h_result, 2);
//             break;
//         }
//         case 4: {
//             float a, b, r;
//             std::memcpy(&a, src1, 4);
//             std::memcpy(&b, src2, 4);
//             r = op(a, b);
//             std::memcpy(dst, &r, 4);
//             break;
//         }
//         case 8: {
//             double a, b, r;
//             std::memcpy(&a, src1, 8);
//             std::memcpy(&b, src2, 8);
//             r = op(a, b);
//             std::memcpy(dst, &r, 8);
//             break;
//         }
//         default:
//             assert(0 && "Unsupported data size for floating point");
//         }
//     } else {
//         // 整数运算
//         if (is_signed) {
//             // 有符号整数
//             switch (bytes) {
//             case 1: {
//                 int8_t a, b, r;
//                 std::memcpy(&a, src1, 1);
//                 std::memcpy(&b, src2, 1);
//                 r = op(a, b);

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = (r < 0);
//                     // 溢出：两个正数相加得负数，或两个负数相加得正数
//                     context->cc_reg.overflow = ((a > 0 && b > 0 && r < 0) ||
//                                                 (a < 0 && b < 0 && r > 0));
//                     // 有符号数的进位标志通常不使用，但这里设置为溢出标志
//                     context->cc_reg.carry = context->cc_reg.overflow;
//                 }

//                 std::memcpy(dst, &r, 1);
//                 break;
//             }
//             case 2: {
//                 int16_t a, b, r;
//                 std::memcpy(&a, src1, 2);
//                 std::memcpy(&b, src2, 2);
//                 r = op(a, b);

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = (r < 0);
//                     context->cc_reg.overflow = ((a > 0 && b > 0 && r < 0) ||
//                                                 (a < 0 && b < 0 && r > 0));
//                     context->cc_reg.carry = context->cc_reg.overflow;
//                 }

//                 std::memcpy(dst, &r, 2);
//                 break;
//             }
//             case 4: {
//                 int32_t a, b, r;
//                 std::memcpy(&a, src1, 4);
//                 std::memcpy(&b, src2, 4);
//                 r = op(a, b);

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = (r < 0);
//                     context->cc_reg.overflow = ((a > 0 && b > 0 && r < 0) ||
//                                                 (a < 0 && b < 0 && r > 0));
//                     context->cc_reg.carry = context->cc_reg.overflow;
//                 }

//                 std::memcpy(dst, &r, 4);
//                 break;
//             }
//             case 8: {
//                 int64_t a, b, r;
//                 std::memcpy(&a, src1, 8);
//                 std::memcpy(&b, src2, 8);
//                 r = op(a, b);

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = (r < 0);
//                     context->cc_reg.overflow = ((a > 0 && b > 0 && r < 0) ||
//                                                 (a < 0 && b < 0 && r > 0));
//                     context->cc_reg.carry = context->cc_reg.overflow;
//                 }

//                 std::memcpy(dst, &r, 8);
//                 break;
//             }
//             default:
//                 assert(0 && "Unsupported data size for signed integer");
//             }
//         } else {
//             // 无符号整数
//             switch (bytes) {
//             case 1: {
//                 uint8_t a, b, r;
//                 std::memcpy(&a, src1, 1);
//                 std::memcpy(&b, src2, 1);
//                 // 使用更大的类型检测溢出
//                 uint16_t temp_result = op((uint16_t)a, (uint16_t)b);
//                 r = (uint8_t)temp_result;

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = false; // 无符号数没有符号标志
//                     context->cc_reg.overflow = false; // 无符号数没有溢出标志
//                     context->cc_reg.carry = (temp_result > UINT8_MAX);
//                 }

//                 std::memcpy(dst, &r, 1);
//                 break;
//             }
//             case 2: {
//                 uint16_t a, b, r;
//                 std::memcpy(&a, src1, 2);
//                 std::memcpy(&b, src2, 2);
//                 uint32_t temp_result = op((uint32_t)a, (uint32_t)b);
//                 r = (uint16_t)temp_result;

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = false;
//                     context->cc_reg.overflow = false;
//                     context->cc_reg.carry = (temp_result > UINT16_MAX);
//                 }

//                 std::memcpy(dst, &r, 2);
//                 break;
//             }
//             case 4: {
//                 uint32_t a, b, r;
//                 std::memcpy(&a, src1, 4);
//                 std::memcpy(&b, src2, 4);
//                 uint64_t temp_result = op((uint64_t)a, (uint64_t)b);
//                 r = (uint32_t)temp_result;

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = false;
//                     context->cc_reg.overflow = false;
//                     context->cc_reg.carry = (temp_result > UINT32_MAX);
//                 }

//                 std::memcpy(dst, &r, 4);
//                 break;
//             }
//             case 8: {
//                 uint64_t a, b, r;
//                 std::memcpy(&a, src1, 8);
//                 std::memcpy(&b, src2, 8);
//                 // 使用128位类型检测溢出
//                 __uint128_t temp_result = op((__uint128_t)a, (__uint128_t)b);
//                 r = (uint64_t)temp_result;

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = false;
//                     context->cc_reg.overflow = false;
//                     context->cc_reg.carry = (temp_result > UINT64_MAX);
//                 }

//                 std::memcpy(dst, &r, 8);
//                 break;
//             }
//             default:
//                 assert(0 && "Unsupported data size for unsigned integer");
//             }
//         }
//     }
// }

// // 通用模板函数，用于处理一元算术操作
// template <typename OpFunc>
// void process_unary_arithmetic(ThreadContext *context, void *dst, void *src,
//                               int bytes, bool is_float, bool is_signed,
//                               bool update_cc, OpFunc op) {
//     if (is_float) {
//         // 浮点运算
//         switch (bytes) {
//         case 2: {
//             // 需要 f16 支持（简化：转 f32 计算）
//             uint16_t h;
//             std::memcpy(&h, src, 2);

//             float f = f16_to_f32(h);
//             float result = op(f);
//             uint16_t h_result = f32_to_f16(result);

//             std::memcpy(dst, &h_result, 2);
//             break;
//         }
//         case 4: {
//             float a, r;
//             std::memcpy(&a, src, 4);
//             r = op(a);
//             std::memcpy(dst, &r, 4);
//             break;
//         }
//         case 8: {
//             double a, r;
//             std::memcpy(&a, src, 8);
//             r = op(a);
//             std::memcpy(dst, &r, 8);
//             break;
//         }
//         default:
//             assert(0 && "Unsupported data size for floating point");
//         }
//     } else {
//         // 整数运算
//         if (is_signed) {
//             // 有符号整数
//             switch (bytes) {
//             case 1: {
//                 int8_t a, r;
//                 std::memcpy(&a, src, 1);
//                 r = op(a);

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = (r < 0);
//                     context->cc_reg.overflow = false; //
//                     一元运算通常不涉及溢出 context->cc_reg.carry = false; //
//                     一元运算通常不设置进位
//                 }

//                 std::memcpy(dst, &r, 1);
//                 break;
//             }
//             case 2: {
//                 int16_t a, r;
//                 std::memcpy(&a, src, 2);
//                 r = op(a);

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = (r < 0);
//                     context->cc_reg.overflow = false;
//                     context->cc_reg.carry = false;
//                 }

//                 std::memcpy(dst, &r, 2);
//                 break;
//             }
//             case 4: {
//                 int32_t a, r;
//                 std::memcpy(&a, src, 4);
//                 r = op(a);

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = (r < 0);
//                     context->cc_reg.overflow = false;
//                     context->cc_reg.carry = false;
//                 }

//                 std::memcpy(dst, &r, 4);
//                 break;
//             }
//             case 8: {
//                 int64_t a, r;
//                 std::memcpy(&a, src, 8);
//                 r = op(a);

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = (r < 0);
//                     context->cc_reg.overflow = false;
//                     context->cc_reg.carry = false;
//                 }

//                 std::memcpy(dst, &r, 8);
//                 break;
//             }
//             default:
//                 assert(0 && "Unsupported data size for signed integer");
//             }
//         } else {
//             // 无符号整数
//             switch (bytes) {
//             case 1: {
//                 uint8_t a, r;
//                 std::memcpy(&a, src, 1);
//                 r = op(a);

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = false; // 无符号数没有符号标志
//                     context->cc_reg.overflow = false;
//                     context->cc_reg.carry = false;
//                 }

//                 std::memcpy(dst, &r, 1);
//                 break;
//             }
//             case 2: {
//                 uint16_t a, r;
//                 std::memcpy(&a, src, 2);
//                 r = op(a);

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = false;
//                     context->cc_reg.overflow = false;
//                     context->cc_reg.carry = false;
//                 }

//                 std::memcpy(dst, &r, 2);
//                 break;
//             }
//             case 4: {
//                 uint32_t a, r;
//                 std::memcpy(&a, src, 4);
//                 r = op(a);

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = false;
//                     context->cc_reg.overflow = false;
//                     context->cc_reg.carry = false;
//                 }

//                 std::memcpy(dst, &r, 4);
//                 break;
//             }
//             case 8: {
//                 uint64_t a, r;
//                 std::memcpy(&a, src, 8);
//                 r = op(a);

//                 // 如果有.cc修饰符，则更新条件码寄存器
//                 if (update_cc) {
//                     context->cc_reg.zero = (r == 0);
//                     context->cc_reg.sign = false;
//                     context->cc_reg.overflow = false;
//                     context->cc_reg.carry = false;
//                 }

//                 std::memcpy(dst, &r, 8);
//                 break;
//             }
//             default:
//                 assert(0 && "Unsupported data size for unsigned integer");
//             }
//         }
//     }
// }

void ADD_Handler::process_operation(ThreadContext *context, void *op[3],
                            const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];

    // 检查是否存在.cc修饰符，决定是否更新条件码寄存器
    bool update_cc = hasCCQualifier(qualifiers);

    // 使用模板函数执行加法操作
    process_binary_arithmetic(context, dst, src1, src2, bytes, is_float,
                              is_signed, update_cc,
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

    // 检查是否存在.cc修饰符，决定是否更新条件码寄存器
    bool update_cc = hasCCQualifier(qualifiers);

    // 使用模板函数执行减法操作
    process_binary_arithmetic(context, dst, src1, src2, bytes, is_float,
                              is_signed, update_cc,
                              [](auto a, auto b) { return a - b; });
}

void NEG::process_operation(ThreadContext *context, void *op[2],
                            const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src = op[1];

    // 检查是否存在.cc修饰符，决定是否更新条件码寄存器
    bool update_cc = hasCCQualifier(qualifiers);

    // 使用模板函数执行取负操作
    process_unary_arithmetic(context, dst, src, bytes, is_float, is_signed,
                             update_cc, [](auto val) { return -val; });
}

void ABS::process_operation(ThreadContext *context, void *op[2],
                            const std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    bool is_signed = TypeUtils::is_signed_type(qualifiers);
    void *dst = op[0];
    void *src = op[1];

    // 检查是否存在.cc修饰符，决定是否更新条件码寄存器
    bool update_cc = hasCCQualifier(qualifiers);

    // 使用模板函数执行绝对值操作
    process_unary_arithmetic(
        context, dst, src, bytes, is_float, is_signed, update_cc, [](auto val) {
            if constexpr (std::is_floating_point_v<
                              std::decay_t<decltype(val)>>) {
                return std::abs(val);
            } else if constexpr (std::is_signed_v<
                                     std::decay_t<decltype(val)>>) {
                return val < 0 ? -val : val;
            } else {
                return val; // 无符号类型直接返回
            }
        });
}
