#ifndef TYPE_UTILS_H
#define TYPE_UTILS_H

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

class ThreadContext;

enum DTYPE { DNONE, DINT, DFLOAT };
namespace TypeUtils {

bool is_float_type(const std::vector<ptxemu::ir::Qualifier> &qualifiers);
ptxemu::ir::Qualifier get_comparison_op(const std::vector<ptxemu::ir::Qualifier> &qualifiers);
bool is_signed_type(const std::vector<ptxemu::ir::Qualifier> &qualifiers);

// 浮点 NaN 检查
inline bool is_nan(float x) { return x != x; }
inline bool is_nan(double x) { return x != x; }
inline bool is_nan(uint16_t h) {
    uint16_t exp = (h >> 10) & 0x1F;
    uint16_t mant = h & 0x3FF;
    return (exp == 0x1F) && (mant != 0);
}

// === 通用比较模板（用于整数和 float/double）===
template <typename T>
inline bool compare(const T &a, const T &b, ptxemu::ir::Qualifier op) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        if (is_nan(a) || is_nan(b)) {
            return (op == ptxemu::ir::Qualifier::Q_NE);
        }
    }
    switch (op) {
    case ptxemu::ir::Qualifier::Q_EQ:
        return a == b;
    case ptxemu::ir::Qualifier::Q_NE:
        return a != b;
    case ptxemu::ir::Qualifier::Q_LT:
        return a < b;
    case ptxemu::ir::Qualifier::Q_LE:
        return a <= b;
    case ptxemu::ir::Qualifier::Q_GT:
        return a > b;
    case ptxemu::ir::Qualifier::Q_GE:
        return a >= b;
    default:
        return false;
    }
}

template <>
inline bool compare<uint16_t>(const uint16_t &a, const uint16_t &b,
                              ptxemu::ir::Qualifier op) {
    // uint16_t 应作为无符号整数进行比较，不转换为 float
    switch (op) {
    case ptxemu::ir::Qualifier::Q_EQ:
        return a == b;
    case ptxemu::ir::Qualifier::Q_NE:
        return a != b;
    case ptxemu::ir::Qualifier::Q_LT:
        return a < b; // 无符号比较
    case ptxemu::ir::Qualifier::Q_LE:
        return a <= b;
    case ptxemu::ir::Qualifier::Q_GT:
        return a > b;
    case ptxemu::ir::Qualifier::Q_GE:
        return a >= b;
    default:
        return false;
    }
}

#define DISPATCH(Q, T, op)                                                     \
    case Q: {                                                                  \
        T a, b;                                                                \
        std::memcpy(&a, src1, sizeof(T));                                      \
        std::memcpy(&b, src2, sizeof(T));                                      \
        __result = TypeUtils::compare(a, b, op) ? 1 : 0;                       \
        break;                                                                 \
    }

#define DISPATCH_F16(Q, op)                                                    \
    case Q: {                                                                  \
        uint16_t a, b;                                                         \
        std::memcpy(&a, src1, 2);                                              \
        std::memcpy(&b, src2, 2);                                              \
        __result = TypeUtils::compare(a, b, op) ? 1 : 0;                       \
        break;                                                                 \
    }

#define SET_P_COMPARE(op, dtype, to, src1, src2)                               \
    do {                                                                       \
        uint8_t __result = 0;                                                  \
        switch (dtype) {                                                       \
            DISPATCH(ptxemu::ir::Qualifier::Q_U8, uint8_t, op)                             \
            DISPATCH(ptxemu::ir::Qualifier::Q_S8, int8_t, op)                              \
            DISPATCH(ptxemu::ir::Qualifier::Q_B8, uint8_t, op)                             \
            DISPATCH(ptxemu::ir::Qualifier::Q_U16, uint16_t, op)                           \
            DISPATCH(ptxemu::ir::Qualifier::Q_S16, int16_t, op)                            \
            DISPATCH(ptxemu::ir::Qualifier::Q_B16, uint16_t, op)                           \
            DISPATCH_F16(ptxemu::ir::Qualifier::Q_F16, op)                                 \
            DISPATCH(ptxemu::ir::Qualifier::Q_U32, uint32_t, op)                           \
            DISPATCH(ptxemu::ir::Qualifier::Q_S32, int32_t, op)                            \
            DISPATCH(ptxemu::ir::Qualifier::Q_B32, uint32_t, op)                           \
            DISPATCH(ptxemu::ir::Qualifier::Q_F32, float, op)                              \
            DISPATCH(ptxemu::ir::Qualifier::Q_U64, uint64_t, op)                           \
            DISPATCH(ptxemu::ir::Qualifier::Q_S64, int64_t, op)                            \
            DISPATCH(ptxemu::ir::Qualifier::Q_B64, uint64_t, op)                           \
            DISPATCH(ptxemu::ir::Qualifier::Q_F64, double, op)                             \
            DISPATCH(ptxemu::ir::Qualifier::Q_PRED, uint8_t, op)                           \
                                                                               \
        default:                                                               \
            __result = 0;                                                      \
        }                                                                      \
        *static_cast<uint8_t *>(to) = __result;                                \
    } while (0)
} // namespace TypeUtils

#endif // TYPE_UTILS_H