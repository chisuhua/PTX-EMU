#include "ptxsim/instruction_handlers.h"
#include "ptxsim/utils/macros.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

// 通用模板函数，用于处理二元位运算操作
template<typename OpFunc>
void process_binary_bitwise(void *dst, void *src1, void *src2, int bytes, OpFunc op) {
    switch (bytes) {
    case 1:
        *(uint8_t *)dst = op(*(uint8_t *)src1, *(uint8_t *)src2);
        break;
    case 2:
        *(uint16_t *)dst = op(*(uint16_t *)src1, *(uint16_t *)src2);
        break;
    case 4:
        *(uint32_t *)dst = op(*(uint32_t *)src1, *(uint32_t *)src2);
        break;
    case 8:
        *(uint64_t *)dst = op(*(uint64_t *)src1, *(uint64_t *)src2);
        break;
    default:
        UNSUPPORTED_TYPESIZE("bitwise operation");
    }
}

void AndHandler::processOperation(ThreadContext *context, void **operands,
                            const std::vector<Qualifier> &qualifiers,
                            const std::vector<char> *operand_is_immediate) {
    int bytes = getBytes(qualifiers);
    void *dst = operands[0];
    void *src1 = operands[1];
    void *src2 = operands[2];

    process_binary_bitwise(dst, src1, src2, bytes, [](uint64_t a, uint64_t b) { return a & b; });
}

void OrHandler::processOperation(ThreadContext *context, void **operands,
                           const std::vector<Qualifier> &qualifiers,
                           const std::vector<char> *operand_is_immediate) {
    void *dst = operands[0];
    void *src1 = operands[1];
    void *src2 = operands[2];
    int bytes = getBytes(qualifiers);

    process_binary_bitwise(dst, src1, src2, bytes, [](uint64_t a, uint64_t b) { return a | b; });
}

void XorHandler::processOperation(ThreadContext *context, void **operands,
                            const std::vector<Qualifier> &qualifiers,
                            const std::vector<char> *operand_is_immediate) {
    void *dst = operands[0];
    void *src1 = operands[1];
    void *src2 = operands[2];
    int bytes = getBytes(qualifiers);

    process_binary_bitwise(dst, src1, src2, bytes, [](uint64_t a, uint64_t b) { return a ^ b; });
}

// 通用模板函数，用于处理移位操作
template<typename OpFunc>
void process_shift_operation(void *dst, void *src1, void *src2, int bytes, OpFunc op) {
    switch (bytes) {
    case 1: {
        *(uint8_t *)dst = op(*(uint8_t *)src1, *(uint8_t *)src2);
        break;
    }
    case 2: {
        *(uint16_t *)dst = op(*(uint16_t *)src1, *(uint16_t *)src2);
        break;
    }
    case 4: {
        *(uint32_t *)dst = op(*(uint32_t *)src1, *(uint32_t *)src2);
        break;
    }
    case 8: {
        *(uint64_t *)dst = op(*(uint64_t *)src1, *(uint64_t *)src2);
        break;
    }
    default:
        UNSUPPORTED_TYPESIZE("shift operation");
    }
}

void ShlHandler::processOperation(ThreadContext *context, void **operands,
                            const std::vector<Qualifier> &qualifiers,
                            const std::vector<char> *operand_is_immediate) {
    void *dst = operands[0];
    void *src1 = operands[1];
    void *src2 = operands[2];
    int bytes = getBytes(qualifiers);

    process_shift_operation(dst, src1, src2, bytes, 
        [](uint64_t a, uint64_t b) { return a << b; });
}

void ShrHandler::processOperation(ThreadContext *context, void **operands,
                            const std::vector<Qualifier> &qualifiers,
                            const std::vector<char> *operand_is_immediate) {
    void *dst = operands[0];
    void *src1 = operands[1];
    void *src2 = operands[2];
    int bytes = getBytes(qualifiers);

    process_shift_operation(dst, src1, src2, bytes,
        [](uint64_t a, uint64_t b) { return a >> b; });
}

// Bit Field Extract (bfe)
//
// Per PTX ISA spec:
//   d = a[b+31:b] for .u32/.s32  (extract `len` bits starting at bit `pos`)
//   d = a[b+63:b] for .u64/.s64
//
// Edge cases (per ISA):
//   - If b+31 (or b+63) exceeds the width, the result is zero for .u*, or
//     sign-extended for .s*.
//   - If b exceeds the width, the result is zero.
//   - If len is zero, the result is zero.
void BfeHandler::processOperation(ThreadContext *context, void **operands,
                            const std::vector<Qualifier> &qualifiers,
                            const std::vector<char> *operand_is_immediate) {
    void *dst = operands[0];
    void *src = operands[1];
    void *pos = operands[2];
    void *len = operands[3];
    int bytes = getBytes(qualifiers);

    if (!dst || !src || !pos || !len) return;

    uint64_t value = 0;
    std::memcpy(&value, src, bytes);
    uint32_t pos_val = 0;
    std::memcpy(&pos_val, pos, bytes);
    uint32_t len_val = 0;
    std::memcpy(&len_val, len, bytes);

    uint32_t total_bits = bytes * 8;
    uint64_t result = 0;

    if (len_val > 0 && pos_val < total_bits) {
        uint32_t effective_len = std::min(len_val, total_bits - pos_val);
        uint64_t mask =
            (effective_len >= 64) ? ~0ULL : ((1ULL << effective_len) - 1);
        result = (value >> pos_val) & mask;

        bool is_signed = false;
        for (auto q : qualifiers) {
            if (q == Qualifier::Q_S32 || q == Qualifier::Q_S64) {
                is_signed = true;
                break;
            }
        }
        if (is_signed && effective_len > 0 && effective_len < total_bits) {
            if (result & (1ULL << (effective_len - 1))) {
                uint64_t sign_mask = ~((1ULL << effective_len) - 1);
                result |= sign_mask;
            }
        }
    }

    std::memcpy(dst, &result, bytes);
}

// 辅助：高效 popcount（使用编译器内置函数）
inline uint32_t popcount_u64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_popcountll(x));
#elif defined(_MSC_VER)
    return __popcnt64(x);
#else
    // 回退：查表法（确保可移植）
    static const uint8_t table[256] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4,
        2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4,
        2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
        4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5,
        3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
        4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};
    uint32_t c = 0;
    for (int i = 0; i < 8; ++i) {
        c += table[x & 0xFF];
        x >>= 8;
    }
    return c;
#endif
}

void PopcHandler::processOperation(ThreadContext *context, void **operands,
                             const std::vector<Qualifier> &qualifiers,
                             const std::vector<char> *operand_is_immediate) {
    void *dst = operands[0];
    void *src1 = operands[1];
    int bytes = getBytes(qualifiers);

    // PTX popc 仅定义于整数/位类型（B* / U* / S*）
    // 浮点类型（F*）为非法，但为健壮性，按位解释
    if (bytes == 0 || bytes > 8) {
        // 未知类型：清零目标
        std::memset(dst, 0, 8); // 安全清零（最多8字节）
        return;
    }

    // 从 src1 读取位模式到 uint64_t
    uint64_t value = 0;
    std::memcpy(&value, src1, bytes);

    // 计算 1 的个数
    uint32_t count = popcount_u64(value);

    // 写回结果（宽度 = bytes，高位自动为0）
    std::memcpy(dst, &count, bytes);
}

inline uint32_t clz_u64(uint64_t x, size_t width) {
    if (x == 0) {
        return static_cast<uint32_t>(width * 8);
    }

#if defined(__GNUC__) || defined(__clang__)
    if (width == 8) {
        return static_cast<uint32_t>(__builtin_clzll(x));
    } else if (width == 4) {
        return static_cast<uint32_t>(__builtin_clz(static_cast<uint32_t>(x)));
    } else if (width == 2) {
        return static_cast<uint32_t>(
                   __builtin_clz(static_cast<uint32_t>(x << 16))) -
               16;
    } else { // width == 1
        return static_cast<uint32_t>(
                   __builtin_clz(static_cast<uint32_t>(x << 24))) -
               24;
    }
#else
    uint32_t total_bits = static_cast<uint32_t>(width * 8);
    for (uint32_t i = 0; i < total_bits; ++i) {
        if (x & (static_cast<uint64_t>(1) << (total_bits - 1 - i))) {
            return i;
        }
    }
    return total_bits;
#endif
}

void ClzHandler::processOperation(ThreadContext *context, void **operands,
                            const std::vector<Qualifier> &qualifiers,
                            const std::vector<char> *operand_is_immediate) {
    void *dst = operands[0];
    void *src1 = operands[1];
    int bytes = getBytes(qualifiers);
    if (bytes == 0 || bytes > 8) {
        std::memset(dst, 0, 8); // 安全清零
        return;
    }

    // 读取位模式（按无符号解释）
    uint64_t value = 0;
    std::memcpy(&value, src1, bytes);

    // 计算 CLZ
    uint32_t result = clz_u64(value, bytes);

    // 写回（宽度 = bytes）
    std::memcpy(dst, &result, bytes);
}

// 通用模板函数，用于处理一元位运算操作
template<typename OpFunc>
void process_unary_bitwise(void *dst, void *src, int bytes, OpFunc op) {
    switch (bytes) {
    case 1:
        *(uint8_t *)dst = op(*(uint8_t *)src);
        break;
    case 2:
        *(uint16_t *)dst = op(*(uint16_t *)src);
        break;
    case 4:
        *(uint32_t *)dst = op(*(uint32_t *)src);
        break;
    case 8:
        *(uint64_t *)dst = op(*(uint64_t *)src);
        break;
    default:
        UNSUPPORTED_TYPESIZE("unary bitwise operation");
    }
}

void NotHandler::processOperation(ThreadContext *context, void **operands,
                            const std::vector<Qualifier> &qualifiers,
                            const std::vector<char> *operand_is_immediate) {
    void *dst = operands[0];
    void *src = operands[1];
    int bytes = getBytes(qualifiers);

    process_unary_bitwise(dst, src, bytes, [](uint64_t x) { return ~x; });
}
