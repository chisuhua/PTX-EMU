#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

// 半精度浮点数转换函数声明
inline float f16_to_f32(uint16_t h);
inline uint16_t f32_to_f16(float f);

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

void SQRT_Handler::processOperation(ThreadContext *context, void *op[2],
                    const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    process_unary_math(dst, src, bytes, is_float, [](auto x) { return std::sqrt(x); });
}

void SIN_Handler::processOperation(ThreadContext *context, void *op[2],
                   const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    process_unary_math(dst, src, bytes, is_float, [](auto x) { return std::sin(x); });
}

void COS_Handler::processOperation(ThreadContext *context, void *op[2],
                   const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    process_unary_math(dst, src, bytes, is_float, [](auto x) { return std::cos(x); });
}

void RCP_Handler::processOperation(ThreadContext *context, void *op[2],
                   const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // RCP指令只支持浮点类型
    assert(is_float && "RCP instruction only supports floating point types");

    process_unary_math(dst, src, bytes, is_float, [](auto x) { return 1.0 / x; });
}

void LG2_Handler::processOperation(ThreadContext *context, void *op[2],
                   const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    assert(is_float && "LG2 instruction only supports floating point types");
    
    process_unary_math(dst, src, bytes, is_float, [](auto x) { return std::log2(x); });
}

void EX2_Handler::processOperation(ThreadContext *context, void *op[2],
                   const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    assert(is_float && "EX2 instruction only supports floating point types");
    
    process_unary_math(dst, src, bytes, is_float, [](auto x) { return std::exp2(x); });
}

void RSQRT_Handler::processOperation(ThreadContext *context, void *op[2],
                     const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = getBytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    assert(is_float && "RSQRT instruction only supports floating point types");
    
    process_unary_math(dst, src, bytes, is_float, [](auto x) { return 1.0 / std::sqrt(x); });
}