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
        case 1: {
            int8_t v;
            if (value > 127)
                v = 127;
            else if (value < -128)
                v = -128;
            else
                v = (int8_t)value;
            // Sign-extend int8 to full 32-bit register
            *(int32_t *)dst = (int32_t)v;
            return;
        }
        case 2: {
            int16_t v;
            if (value > 32767)
                v = 32767;
            else if (value < -32768)
                v = -32768;
            else
                v = (int16_t)value;
            // Sign-extend int16 to full 32-bit register
            *(int32_t *)dst = (int32_t)v;
            return;
        }
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
            switch (dst_bytes) {
            case 1:
                *(uint32_t *)dst = 0;
                return;
            case 2:
                *(uint32_t *)dst = 0;
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
        case 1: {
            uint8_t v = (value > 255) ? (uint8_t)255 : (uint8_t)value;
            // Zero-extend uint8 to full 32-bit register
            *(uint32_t *)dst = (uint32_t)v;
            return;
        }
        case 2: {
            uint16_t v = (value > 65535) ? (uint16_t)65535 : (uint16_t)value;
            // Zero-extend uint16 to full 32-bit register
            *(uint32_t *)dst = (uint32_t)v;
            return;
        }
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
// 注意: PTX 寄存器总是 32-bit (b8/b16 类型自动零扩展/符号扩展到 32-bit 寄存器)
// 所以 s8/s16 结果必须符号扩展 (signed) 或零扩展 (unsigned) 到 full register
static void write_trunc_dst(void *dst, int64_t value, int dst_bytes,
                            bool dst_signed) {
    if (dst_signed) {
        switch (dst_bytes) {
        case 1: {
            int8_t v = (int8_t)value;
            // Sign-extend int8 to full 32-bit register
            *(int32_t *)dst = (int32_t)v;
            return;
        }
        case 2: {
            int16_t v = (int16_t)value;
            // Sign-extend int16 to full 32-bit register
            *(int32_t *)dst = (int32_t)v;
            return;
        }
        case 4:
            *(int32_t *)dst = (int32_t)value;
            return;
        case 8:
        default:
            *(int64_t *)dst = value;
            return;
        }
    } else {
        switch (dst_bytes) {
        case 1: {
            uint8_t v = (uint8_t)value;
            // Zero-extend uint8 to full 32-bit register
            *(uint32_t *)dst = (uint32_t)v;
            return;
        }
        case 2: {
            uint16_t v = (uint16_t)value;
            // Zero-extend uint16 to full 32-bit register
            *(uint32_t *)dst = (uint32_t)v;
            return;
        }
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
