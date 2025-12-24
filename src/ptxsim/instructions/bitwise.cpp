#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

void AND::process_operation(ThreadContext *context, void *op[3],
                            const std::vector<Qualifier> &qualifiers) {
    int bytes = getBytes(qualifiers);
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];

    switch (bytes) {
    case 1:
        *(uint8_t *)dst = (*(uint8_t *)src1) & (*(uint8_t *)src2);
        break;
    case 2:
        *(uint16_t *)dst = (*(uint16_t *)src1) & (*(uint16_t *)src2);
        break;
    case 4:
        *(uint32_t *)dst = (*(uint32_t *)src1) & (*(uint32_t *)src2);
        break;
    case 8:
        *(uint64_t *)dst = (*(uint64_t *)src1) & (*(uint64_t *)src2);
        break;
    default:
        assert(0 && "Unsupported data size for AND operation");
    }
}

void OR::process_operation(ThreadContext *context, void *op[3],
                           const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];
    int bytes = getBytes(qualifiers);

    switch (bytes) {
    case 1:
        *(uint8_t *)dst = (*(uint8_t *)src1) | (*(uint8_t *)src2);
        break;
    case 2:
        *(uint16_t *)dst = (*(uint16_t *)src1) | (*(uint16_t *)src2);
        break;
    case 4:
        *(uint32_t *)dst = (*(uint32_t *)src1) | (*(uint32_t *)src2);
        break;
    case 8:
        *(uint64_t *)dst = (*(uint64_t *)src1) | (*(uint64_t *)src2);
        break;
    default:
        assert(0 && "Unsupported data size for OR operation");
    }
}

void XOR::process_operation(ThreadContext *context, void *op[3],
                            const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];
    int bytes = getBytes(qualifiers);

    switch (bytes) {
    case 1:
        *(uint8_t *)dst = (*(uint8_t *)src1) ^ (*(uint8_t *)src2);
        break;
    case 2:
        *(uint16_t *)dst = (*(uint16_t *)src1) ^ (*(uint16_t *)src2);
        break;
    case 4:
        *(uint32_t *)dst = (*(uint32_t *)src1) ^ (*(uint32_t *)src2);
        break;
    case 8:
        *(uint64_t *)dst = (*(uint64_t *)src1) ^ (*(uint64_t *)src2);
        break;
    default:
        assert(0 && "Unsupported data size for XOR operation");
    }
}

void SHL::process_operation(ThreadContext *context, void *op[3],
                            const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];
    int bytes = getBytes(qualifiers);

    // 根据数据类型执行左移操作
    switch (bytes) {
    case 1: {
        *(uint8_t *)dst = (*(uint8_t *)src1) << (*(uint8_t *)src2);
        break;
    }
    case 2: {
        *(uint16_t *)dst = (*(uint16_t *)src1) << (*(uint16_t *)src2);
        break;
    }
    case 4: {
        *(uint32_t *)dst = (*(uint32_t *)src1) << (*(uint32_t *)src2);
        break;
    }
    case 8: {
        *(uint64_t *)dst = (*(uint64_t *)src1) << (*(uint64_t *)src2);
        break;
    }
    default:
        // 不支持的数据大小
        assert(0 && "Unsupported data size for SHL instruction");
    }
}

void SHR::process_operation(ThreadContext *context, void *op[3],
                            const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];
    int bytes = getBytes(qualifiers);

    // 根据数据类型执行逻辑右移操作（使用无符号类型）
    switch (bytes) {
    case 1: {
        *(uint8_t *)dst = (*(uint8_t *)src1) >> (*(uint8_t *)src2);
        break;
    }
    case 2: {
        *(uint16_t *)dst = (*(uint16_t *)src1) >> (*(uint16_t *)src2);
        break;
    }
    case 4: {
        *(uint32_t *)dst = (*(uint32_t *)src1) >> (*(uint32_t *)src2);
        break;
    }
    case 8: {
        *(uint64_t *)dst = (*(uint64_t *)src1) >> (*(uint64_t *)src2);
        break;
    }
    default:
        // 不支持的数据大小
        assert(0 && "Unsupported data size for SHR instruction");
    }
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

void POPC::process_operation(ThreadContext *context, void *op[2],
                             const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src1 = op[1];
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

void CLZ::process_operation(ThreadContext *context, void *op[2],
                            const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src1 = op[1];
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

void NOT::process_operation(ThreadContext *context, void *op[2],
                            const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src1 = op[1];
    // TODO
}