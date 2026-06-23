// cvt_float_to_float.cpp
// =============================================================================
// FloatToFloatStrategy implementation (T2-6 Sub-task 4a)
//
// 处理 dst_is_float && src_is_float 的所有 PTX CVT 组合:
//   - f32→f32 (identity + .sat NaN→0)
//   - f64→f64 (identity, direct bit copy 不经 float)
//   - f32→f64 / f64→f32 (标量 widening/narrowing)
//   - f16→f16 (identity for half)
//   - f16→f32 / f32→f16 (half ↔ float, via half_utils)
//   - f16→f64 / f64→f16 (half ↔ double, via half_utils → float → double)
//
// 注: src=f64 需用 double 中间值保持精度，f64→f64 不经 float。
// =============================================================================

#include "ptxsim/instructions/cvt/cvt_float_to_float.h"
#include "ptxsim/instructions/cvt/cvt_helpers.h"
#include "ptxsim/utils/half_utils.h"

#include <cmath>

namespace ptxsim {
namespace cvt_strategy {

void FloatToFloatStrategy::convert(void *dst, void *src,
                                   const CvtContext &ctx) const {
    // Special case: f64 → f64 must be bit-exact (no float intermediate).
    // PTX f64 ↔ f64 conversion is identity at the bit level.
    if (ctx.src_bytes == 8 && ctx.dst_bytes == 8) {
        *(double *)dst = *(double *)src;
        return;
    }

    // Special case: f64 → f32 (narrow via static_cast, but check .sat first).
    if (ctx.src_bytes == 8 && ctx.dst_bytes == 4) {
        double src_d = *(double *)src;
        if (ctx.has_sat && std::isnan(src_d)) {
            *(float *)dst = 0.0f;
        } else {
            *(float *)dst = static_cast<float>(src_d);
        }
        return;
    }

    // Special case: src is f16 (decode to float intermediate).
    if (ctx.src_is_half) {
        uint16_t h = *reinterpret_cast<uint16_t *>(src);
        float temp = cvt_helpers::half_to_float(h);
        if (ctx.has_sat && std::isnan(temp)) {
            *(float *)dst = 0.0f;
            return;
        }
        // Dispatch on dst.
        if (ctx.dst_is_half) {
            *(uint16_t *)dst = *(uint16_t *)src; // half→half bit copy
        } else if (ctx.dst_bytes == 4) {
            *(float *)dst = temp;
        } else {
            // dst is f64
            *(double *)dst = static_cast<double>(temp);
        }
        return;
    }

    // Default: src is f32.
    float temp = *(float *)src;
    if (ctx.has_sat) {
        if (std::isnan(temp)) {
            *(float *)dst = 0.0f;
        } else {
            *(float *)dst = temp;
        }
        return;
    }
    if (ctx.dst_bytes == 4) {
        *(float *)dst = temp;
    } else if (ctx.dst_bytes == 2) {
        // dst is f16
        *(uint16_t *)dst = cvt_helpers::float_to_half(temp);
    } else {
        // dst is f64
        *(double *)dst = static_cast<double>(temp);
    }
}

} // namespace cvt_strategy
} // namespace ptxsim
