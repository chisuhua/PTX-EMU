// cvt_int_to_int.cpp
// =============================================================================
// IntToIntStrategy implementation (T2-6 Sub-task 4d)
//
// 4×4 维度矩阵处理:
//   src_bytes ∈ {1, 2, 4, 8} × dst_bytes ∈ {1, 2, 4, 8}
//   + (src_is_signed, dst_is_signed) ∈ {ss, su, us, uu}
//   + .sat 饱和 + 5 舍入模式
//
// 策略: 单一 convert() 方法, 64 组合 (4*4*4) 用嵌套 if-else 展开。
// 复杂度高但每个分支逻辑简单 (截断/符号扩展 + .sat clamp + 舍入)。
// =============================================================================

#include "ptxsim/instructions/cvt/cvt_int_to_int.h"
#include "ptxsim/instructions/cvt/cvt_helpers.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace ptxsim {
namespace cvt_strategy {

// 模板辅助: 读取 src 作为 int64_t (总是有符号)
static int64_t read_src_as_i64(void *src, int src_bytes, bool src_signed) {
    switch (src_bytes) {
    case 1:
        return src_signed ? *(int8_t *)src : (int64_t)(*(uint8_t *)src);
    case 2:
        return src_signed ? *(int16_t *)src : (int64_t)(*(uint16_t *)src);
    case 4:
        return src_signed ? *(int32_t *)src : (int64_t)(*(uint32_t *)src);
    case 8:
    default:
        return src_signed ? *(int64_t *)src : (int64_t)(*(uint64_t *)src);
    }
}

// 模板辅助: 应用 .sat clamp 并写到 dst
// value 已是有符号 int64_t; dst_signed + dst_bytes 决定范围
static void write_sat_dst(void *dst, int64_t value, int dst_bytes,
                          bool dst_signed) {
    if (dst_signed) {
        switch (dst_bytes) {
        case 1:
            if (value > 127)
                *(int8_t *)dst = 127;
            else if (value < -128)
                *(int8_t *)dst = -128;
            else
                *(int8_t *)dst = (int8_t)value;
            return;
        case 2:
            if (value > 32767)
                *(int16_t *)dst = 32767;
            else if (value < -32768)
                *(int16_t *)dst = -32768;
            else
                *(int16_t *)dst = (int16_t)value;
            return;
        case 4:
            if (value > 2147483647LL)
                *(int32_t *)dst = 2147483647;
            else if (value < -2147483647LL - 1)
                *(int32_t *)dst = -2147483647 - 1;
            else
                *(int32_t *)dst = (int32_t)value;
            return;
        case 8:
        default:
            *(int64_t *)dst = value;
            return;
        }
    } else {
        // dst unsigned: clamp to [0, MAX]
        if (value < 0) {
            // negative → 0 (or per PTX spec for u64 large, but src is int64)
            switch (dst_bytes) {
            case 1:
                *(uint8_t *)dst = 0;
                return;
            case 2:
                *(uint16_t *)dst = 0;
                return;
            case 4:
                *(uint32_t *)dst = 0;
                return;
            case 8:
            default:
                *(uint64_t *)dst = 0;
                return;
            }
        }
        switch (dst_bytes) {
        case 1:
            if (value > 255)
                *(uint8_t *)dst = 255;
            else
                *(uint8_t *)dst = (uint8_t)value;
            return;
        case 2:
            if (value > 65535)
                *(uint16_t *)dst = 65535;
            else
                *(uint16_t *)dst = (uint16_t)value;
            return;
        case 4:
            if (value > 4294967295LL)
                *(uint32_t *)dst = 4294967295U;
            else
                *(uint32_t *)dst = (uint32_t)value;
            return;
        case 8:
        default:
            *(uint64_t *)dst = (uint64_t)value;
            return;
        }
    }
}

// 模板辅助: 无 .sat, 写 dst (应用 5 舍入或默认 truncation)
static void write_trunc_dst(void *dst, int64_t value, int dst_bytes,
                            bool dst_signed) {
    if (dst_signed) {
        switch (dst_bytes) {
        case 1:
            *(int8_t *)dst = (int8_t)value;
            return;
        case 2:
            *(int16_t *)dst = (int16_t)value;
            return;
        case 4:
            *(int32_t *)dst = (int32_t)value;
            return;
        case 8:
        default:
            *(int64_t *)dst = value;
            return;
        }
    } else {
        // Unsigned: if value < 0 (signed→unsigned cross), behavior is
        // impl-defined PTX without .sat: source value is treated as bit pattern
        switch (dst_bytes) {
        case 1:
            *(uint8_t *)dst = (uint8_t)value;
            return;
        case 2:
            *(uint16_t *)dst = (uint16_t)value;
            return;
        case 4:
            *(uint32_t *)dst = (uint32_t)value;
            return;
        case 8:
        default:
            *(uint64_t *)dst = (uint64_t)value;
            return;
        }
    }
}

void IntToIntStrategy::convert(void *dst, void *src,
                               const CvtContext &ctx) const {
    int64_t value = read_src_as_i64(src, ctx.src_bytes, ctx.src_is_signed);

    if (ctx.has_sat) {
        write_sat_dst(dst, value, ctx.dst_bytes, ctx.dst_is_signed);
        return;
    }

    // 5 rounding modes (only meaningful for int→int, but PTX allows them)
    if (ctx.has_rn || ctx.has_rni) {
        // Round half to even on int (rarely meaningful; passthrough)
        write_trunc_dst(dst, value, ctx.dst_bytes, ctx.dst_is_signed);
    } else if (ctx.has_rz || ctx.has_rzi) {
        write_trunc_dst(dst, value, ctx.dst_bytes, ctx.dst_is_signed);
    } else if (ctx.has_rm || ctx.has_rmi) {
        write_trunc_dst(dst, value, ctx.dst_bytes, ctx.dst_is_signed);
    } else if (ctx.has_rp || ctx.has_rpi) {
        write_trunc_dst(dst, value, ctx.dst_bytes, ctx.dst_is_signed);
    } else if (ctx.has_rna) {
        write_trunc_dst(dst, value, ctx.dst_bytes, ctx.dst_is_signed);
    } else {
        // Default: PTX spec for int→int without rounding is direct bit
        // truncation/extension. (The old switch preserved source value as
        // int64 then bit-casts to dst.)
        write_trunc_dst(dst, value, ctx.dst_bytes, ctx.dst_is_signed);
    }
}

} // namespace cvt_strategy
} // namespace ptxsim
