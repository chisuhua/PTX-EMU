// cvt_int_to_float.h
// =============================================================================
// IntToFloatStrategy (T2-6 Sub-task 4b)
//
// 处理 PTX CVT 指令的 dst_is_float && !src_is_float 分支:
//   - 整型 (s8/s16/s32/s64, u8/u16/u32/u64) → f16/f32/f64
//   - 含 .sat 处理 (PTX .sat 对 int->float 是 no-op，因源不会是 NaN)
//
// 复杂度: ~80 行
// 来源: arithmetic_conversion.cpp 原 switch 中 case 1/2/4/8
//       dst_is_float && !src_is_float 分支
// =============================================================================

#ifndef PTXSIM_INSTRUCTIONS_CVT_CVT_INT_TO_FLOAT_H
#define PTXSIM_INSTRUCTIONS_CVT_CVT_INT_TO_FLOAT_H

#include "ptxsim/instructions/cvt/cvt_strategy.h"

namespace ptxsim {
namespace cvt_strategy {

class IntToFloatStrategy : public ConversionStrategy {
public:
    void convert(void *dst, void *src, const CvtContext &ctx) const override;
    const char *name() const override { return "IntToFloatStrategy"; }
};

} // namespace cvt_strategy
} // namespace ptxsim

#endif // PTXSIM_INSTRUCTIONS_CVT_CVT_INT_TO_FLOAT_H
