// cvt_float_to_float.h
// =============================================================================
// FloatToFloatStrategy (T2-6 Sub-task 4a)
//
// 处理 PTX CVT 指令的 dst_is_float && src_is_float 分支:
//   - f32 → f32  (identity, 含 .sat 处理 NaN → 0)
//   - f64 → f32 / f32 → f64 / f64 → f64  (标量互转)
//   - f16 ↔ f32 / f16 ↔ f64              (half precision)
//   - f16 → f16                          (identity for half)
//
// 复杂度: 30 行 (Sub-task 4a: 最简单的一个策略)
// 来源: arithmetic_conversion.cpp 原 switch 中 case 1/2/4/8 dst_is_float
//       且 src_is_float 的分支逻辑 (T2-6 Step 2 已迁至 cvt_strategy.cpp 的
//       GeneralCvtStrategy::convert())
// =============================================================================

#ifndef PTXSIM_INSTRUCTIONS_CVT_CVT_FLOAT_TO_FLOAT_H
#define PTXSIM_INSTRUCTIONS_CVT_CVT_FLOAT_TO_FLOAT_H

#include "ptxsim/instructions/cvt/cvt_strategy.h"

namespace ptxsim {
namespace cvt_strategy {

class FloatToFloatStrategy : public ConversionStrategy {
public:
    void convert(void *dst, void *src, const CvtContext &ctx) const override;
    const char *name() const override { return "FloatToFloatStrategy"; }
};

} // namespace cvt_strategy
} // namespace ptxsim

#endif // PTXSIM_INSTRUCTIONS_CVT_CVT_FLOAT_TO_FLOAT_H
