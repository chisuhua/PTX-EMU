// cvt_int_to_float.cpp
// =============================================================================
// IntToFloatStrategy implementation (T2-6 Sub-task 4b)
//
// 处理 dst_is_float && !src_is_float 的所有 PTX CVT 组合:
//   - 8/16/32/64 位整型 (signed 或 unsigned) → f16/f32/f64
//   - .sat 在 int->float 路径是 no-op (源非 NaN)
//
// 策略: 中间值选最宽可用 (s64/d64/u64), 然后窄到 dst。
// 简化: f32/f64 路径用 double 中间 (除 s32/u32→f32 直接转)。
// =============================================================================

#include "ptxsim/instructions/cvt/cvt_int_to_float.h"
#include "ptxsim/instructions/cvt/cvt_helpers.h"

namespace ptxsim {
namespace cvt_strategy {

void IntToFloatStrategy::convert(void *dst, void *src,
                                 const CvtContext &ctx) const {
    // Read source as the widest appropriate intermediate.
    // For src_bytes <= 4, we can use double directly.
    // For src_bytes == 8, we read s64 or u64 directly.
    double dval;
    if (ctx.src_bytes == 1) {
        dval = ctx.src_is_signed ? static_cast<double>(*(int8_t *)src)
                                 : static_cast<double>(*(uint8_t *)src);
    } else if (ctx.src_bytes == 2) {
        dval = ctx.src_is_signed ? static_cast<double>(*(int16_t *)src)
                                 : static_cast<double>(*(uint16_t *)src);
    } else if (ctx.src_bytes == 4) {
        dval = ctx.src_is_signed ? static_cast<double>(*(int32_t *)src)
                                 : static_cast<double>(*(uint32_t *)src);
    } else {
        // 8 bytes
        dval = ctx.src_is_signed ? static_cast<double>(*(int64_t *)src)
                                 : static_cast<double>(*(uint64_t *)src);
    }

    // .sat on int->float: no-op (source cannot be NaN).
    (void)ctx.has_sat;

    // Write to dst: float / double / half
    if (ctx.dst_is_half) {
        // half: convert via float
        *(uint16_t *)dst = cvt_helpers::float_to_half(static_cast<float>(dval));
    } else if (ctx.dst_bytes == 4) {
        *(float *)dst = static_cast<float>(dval);
    } else {
        // dst_bytes == 8 (f64)
        *(double *)dst = dval;
    }
}

} // namespace cvt_strategy
} // namespace ptxsim
