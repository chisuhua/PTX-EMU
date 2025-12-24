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

bool is_float_type(const std::vector<Qualifier> &qualifiers);
Qualifier get_comparison_op(const std::vector<Qualifier> &qualifiers);
bool is_signed_type(const std::vector<Qualifier> &qualifiers);

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
inline bool compare(const T &a, const T &b, Qualifier op) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        if (is_nan(a) || is_nan(b)) {
            return (op == Qualifier::Q_NE);
        }
    }
    switch (op) {
    case Qualifier::Q_EQ:
        return a == b;
    case Qualifier::Q_NE:
        return a != b;
    case Qualifier::Q_LT:
        return a < b;
    case Qualifier::Q_LE:
        return a <= b;
    case Qualifier::Q_GT:
        return a > b;
    case Qualifier::Q_GE:
        return a >= b;
    default:
        return false;
    }
}

template <>
inline bool compare<uint16_t>(const uint16_t &a, const uint16_t &b,
                              Qualifier op) {
    if (is_nan(a) || is_nan(b)) {
        return (op == Qualifier::Q_NE);
    }
    auto to_f32 = [](uint16_t h) -> float {
        uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 1;
        uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f32;
        if (exp == 0x1F) {
            f32 = (sign << 31) | (0xFFU << 23) | (mant << 13);
        } else {
            f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
        float res;
        std::memcpy(&res, &f32, 4);
        return res;
    };
    float fa = to_f32(a), fb = to_f32(b);
    // 复用 float 比较逻辑（需定义 _do_compare<float, OP>）
    return compare(fa, fb, op);
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
            DISPATCH(Qualifier::Q_U8, uint8_t, op)                             \
            DISPATCH(Qualifier::Q_S8, int8_t, op)                              \
            DISPATCH(Qualifier::Q_B8, uint8_t, op)                             \
            DISPATCH(Qualifier::Q_U16, uint16_t, op)                           \
            DISPATCH(Qualifier::Q_S16, int16_t, op)                            \
            DISPATCH(Qualifier::Q_B16, uint16_t, op)                           \
            DISPATCH_F16(Qualifier::Q_F16, op)                                 \
            DISPATCH(Qualifier::Q_U32, uint32_t, op)                           \
            DISPATCH(Qualifier::Q_S32, int32_t, op)                            \
            DISPATCH(Qualifier::Q_B32, uint32_t, op)                           \
            DISPATCH(Qualifier::Q_F32, float, op)                              \
            DISPATCH(Qualifier::Q_U64, uint64_t, op)                           \
            DISPATCH(Qualifier::Q_S64, int64_t, op)                            \
            DISPATCH(Qualifier::Q_B64, uint64_t, op)                           \
            DISPATCH(Qualifier::Q_F64, double, op)                             \
            DISPATCH(Qualifier::Q_PRED, uint8_t, op)                           \
                                                                               \
        default:                                                               \
            __result = 0;                                                      \
        }                                                                      \
        *static_cast<uint8_t *>(to) = __result;                                \
    } while (0)
} // namespace TypeUtils

#endif // TYPE_UTILS_H