// cvt_float_to_int.cpp
// =============================================================================
// FloatToIntStrategy implementation (T2-6 Sub-task 4c + 4d)
//
// 处理 !dst_is_float && src_is_float 的所有 PTX CVT 组合:
//   - 5 舍入模式 (.rn/.rz/.rm/.rp/.rna + .rni/.rzi/.rmi/.rpi 别名)
//   - .sat 饱和 (NaN→0, 上下界 clamp, s64/u64 特殊)
//
// 简化策略: 用 double 中间值，dst 写入分支 dispatch by dst_bytes +
// dst_is_signed + has_sat。
// =============================================================================

#include "ptxsim/instructions/cvt/cvt_float_to_int.h"
#include "ptxsim/instructions/cvt/cvt_helpers.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace ptxsim {
namespace cvt_strategy {

// 提取源 float 值 (via half if needed, else direct read)
static double load_source(void *src, const CvtContext &ctx) {
    if (ctx.src_is_half) {
        uint16_t h = *reinterpret_cast<uint16_t *>(src);
        return static_cast<double>(cvt_helpers::half_to_float(h));
    } else if (ctx.src_bytes == 4) {
        return static_cast<double>(*(float *)src);
    } else {
        // f64
        return *(double *)src;
    }
}

// 应用 5 种舍入模式之一到 double 值,返回 rounded int64_t
static int64_t apply_rounding(double val, const CvtContext &ctx) {
    if (ctx.has_rn || ctx.has_rni) {
        // Round to nearest, ties to even
        return static_cast<int64_t>(
            cvt_helpers::round_half_to_even(static_cast<float>(val)));
    } else if (ctx.has_rz || ctx.has_rzi) {
        return static_cast<int64_t>(std::trunc(val));
    } else if (ctx.has_rm || ctx.has_rmi) {
        return static_cast<int64_t>(std::floor(val));
    } else if (ctx.has_rp || ctx.has_rpi) {
        return static_cast<int64_t>(std::ceil(val));
    } else if (ctx.has_rna) {
        return static_cast<int64_t>((val >= 0.0) ? std::floor(val + 0.5)
                                                 : std::ceil(val - 0.5));
    }
    // Default: PTX spec for cvt.*.f* without rounding is truncation.
    return static_cast<int64_t>(std::trunc(val));
}

void FloatToIntStrategy::convert(void *dst, void *src,
                                 const CvtContext &ctx) const {
    double val = load_source(src, ctx);

    // ---- .sat path ----
    if (ctx.has_sat) {
        // NaN → 0 (PTX spec)
        if (std::isnan(val)) {
            if (ctx.dst_bytes == 1) {
                if (ctx.dst_is_signed) {
                    *(int8_t *)dst = 0;
                } else {
                    *(uint8_t *)dst = 0;
                }
            } else if (ctx.dst_bytes == 2) {
                if (ctx.dst_is_signed) {
                    *(int16_t *)dst = 0;
                } else {
                    *(uint16_t *)dst = 0;
                }
            } else if (ctx.dst_bytes == 4) {
                if (ctx.dst_is_signed) {
                    *(int32_t *)dst = 0;
                } else {
                    *(uint32_t *)dst = 0;
                }
            } else {
                if (ctx.dst_is_signed) {
                    *(int64_t *)dst = 0;
                } else {
                    *(uint64_t *)dst = 0;
                }
            }
            return;
        }
        // Apply .sat clamping per dst type
        if (ctx.dst_bytes == 1) {
            if (ctx.dst_is_signed) {
                int8_t v;
                if (val <= -128.0)
                    v = -128;
                else if (val >= 127.0)
                    v = 127;
                else
                    v = static_cast<int8_t>(val);
                // Sign-extend int8 to full 32-bit register
                *(int32_t *)dst = static_cast<int32_t>(v);
            } else {
                uint8_t v;
                if (val <= 0.0)
                    v = 0;
                else if (val >= 255.0)
                    v = 255;
                else
                    v = static_cast<uint8_t>(val);
                // Zero-extend uint8 to full 32-bit register
                *(uint32_t *)dst = static_cast<uint32_t>(v);
            }
        } else if (ctx.dst_bytes == 2) {
            if (ctx.dst_is_signed) {
                int16_t v;
                if (val <= -32768.0)
                    v = -32768;
                else if (val >= 32767.0)
                    v = 32767;
                else
                    v = static_cast<int16_t>(val);
                // Sign-extend int16 to full 32-bit register
                *(int32_t *)dst = static_cast<int32_t>(v);
            } else {
                uint16_t v;
                if (val <= 0.0)
                    v = 0;
                else if (val >= 65535.0)
                    v = 65535;
                else
                    v = static_cast<uint16_t>(val);
                // Zero-extend uint16 to full 32-bit register
                *(uint32_t *)dst = static_cast<uint32_t>(v);
            }
        } else if (ctx.dst_bytes == 4) {
            if (ctx.dst_is_signed) {
                if (val <= -2147483648.0) {
                    *(int32_t *)dst = -2147483647 - 1; // INT32_MIN
                } else if (val >= 2147483647.0) {
                    *(int32_t *)dst = 2147483647;
                } else {
                    *(int32_t *)dst = static_cast<int32_t>(val);
                }
            } else {
                // u32 .sat: special handling due to f32 precision at boundary
                // 4294967295.0f rounds up to 4294967296.0f in float32, so
                // boundary values must be compared in float precision.
                float temp = static_cast<float>(val);
                if (std::isnan(temp)) {
                    *(uint32_t *)dst = 0;
                } else if (temp <= 0.0f) {
                    *(uint32_t *)dst = 0;
                } else if (temp > 4294967295.0f) {
                    *(uint32_t *)dst = 4294967295U;
                } else {
                    *(uint32_t *)dst = static_cast<uint32_t>(temp);
                }
            }
        } else {
            // 8 bytes
            if (ctx.dst_is_signed) {
                if (val >= 9223372036854775807.0) {
                    *(int64_t *)dst = 9223372036854775807LL;
                } else {
                    *(int64_t *)dst = static_cast<int64_t>(val);
                }
            } else {
                if (val <= 0.0) {
                    *(uint64_t *)dst = 0;
                } else if (val > 18446744073709551615.0) {
                    *(uint64_t *)dst = 18446744073709551615ULL;
                } else {
                    *(uint64_t *)dst = static_cast<uint64_t>(val);
                }
            }
        }
        return;
    }

    // ---- non-.sat path: apply rounding then write ----
    int64_t rounded = apply_rounding(val, ctx);

    if (ctx.dst_bytes == 1) {
        if (ctx.dst_is_signed) {
            // Sign-extend int8 to full 32-bit register
            *(int32_t *)dst =
                static_cast<int32_t>(static_cast<int8_t>(rounded));
        } else {
            uint8_t v =
                (rounded < 0) ? uint8_t(0) : static_cast<uint8_t>(rounded);
            // Zero-extend uint8 to full 32-bit register
            *(uint32_t *)dst = static_cast<uint32_t>(v);
        }
    } else if (ctx.dst_bytes == 2) {
        if (ctx.dst_is_signed) {
            // Sign-extend int16 to full 32-bit register
            *(int32_t *)dst =
                static_cast<int32_t>(static_cast<int16_t>(rounded));
        } else {
            uint16_t v =
                (rounded < 0) ? uint16_t(0) : static_cast<uint16_t>(rounded);
            // Zero-extend uint16 to full 32-bit register
            *(uint32_t *)dst = static_cast<uint32_t>(v);
        }
    } else if (ctx.dst_bytes == 4) {
        if (ctx.dst_is_signed) {
            *(int32_t *)dst = static_cast<int32_t>(rounded);
        } else {
            // u32 non-.sat: special handling due to f32 precision at boundary
            if (ctx.has_rn || ctx.has_rni) {
                if (cvt_helpers::should_saturate_uint32(static_cast<float>(val),
                                                        4294967295.5f)) {
                    *(uint32_t *)dst = 4294967295U;
                } else {
                    *(uint32_t *)dst = static_cast<uint32_t>(rounded);
                }
            } else if (ctx.has_rz || ctx.has_rzi) {
                if (cvt_helpers::should_saturate_uint32(static_cast<float>(val),
                                                        4294967296.0f)) {
                    *(uint32_t *)dst = 4294967295U;
                } else {
                    *(uint32_t *)dst = static_cast<uint32_t>(rounded);
                }
            } else if (ctx.has_rm || ctx.has_rmi) {
                if (cvt_helpers::should_saturate_uint32(static_cast<float>(val),
                                                        4294967296.0f)) {
                    *(uint32_t *)dst = 4294967295U;
                } else {
                    *(uint32_t *)dst = static_cast<uint32_t>(rounded);
                }
            } else if (ctx.has_rp || ctx.has_rpi) {
                if (cvt_helpers::should_saturate_uint32(static_cast<float>(val),
                                                        4294967295.0f)) {
                    *(uint32_t *)dst = 4294967295U;
                } else {
                    *(uint32_t *)dst = static_cast<uint32_t>(rounded);
                }
            } else {
                if (rounded < 0) {
                    *(uint32_t *)dst = 0;
                } else {
                    *(uint32_t *)dst = static_cast<uint32_t>(rounded);
                }
            }
        }
    } else { // 8 bytes
        if (ctx.dst_is_signed) {
            *(int64_t *)dst = rounded;
        } else {
            if (rounded < 0) {
                *(uint64_t *)dst = 0;
            } else {
                *(uint64_t *)dst = static_cast<uint64_t>(rounded);
            }
        }
    }
}

} // namespace cvt_strategy
} // namespace ptxsim
