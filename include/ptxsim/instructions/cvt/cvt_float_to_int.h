// cvt_float_to_int.h
// =============================================================================
// FloatToIntStrategy (T2-6 Sub-task 4c + 4d)
//
// 处理 PTX CVT 指令的 !dst_is_float && src_is_float 分支:
//   - f16/f32/f64 → s8/s16/s32/s64/u8/u16/u32/u64
//   - 含 .sat 饱和处理 (NaN→0, 上界/下界 clamp, s64/u64 特殊)
//   - 5 种舍入模式 (.rn/.rz/.rm/.rp/.rna + .rni/.rzi/.rmi/.rpi 别名)
//
// 复杂度: ~180 行 (Sub-task 4d: 最复杂的一个策略)
// 来源: arithmetic_conversion.cpp 原 switch 中 case 1/2/4/8
//       !dst_is_float && src_is_float 分支
//
// 已知 bug 修复: P1-4.1 - f32→s32 / f64→s64 路径补 r2 写入 (Sub-task 5)
// =============================================================================

#ifndef PTXSIM_INSTRUCTIONS_CVT_CVT_FLOAT_TO_INT_H
#define PTXSIM_INSTRUCTIONS_CVT_CVT_FLOAT_TO_INT_H

#include "ptxsim/instructions/cvt/cvt_strategy.h"

namespace ptxsim {
namespace cvt_strategy {

class FloatToIntStrategy : public ConversionStrategy {
public:
    void convert(void *dst, void *src, const CvtContext &ctx) const override;
    const char *name() const override { return "FloatToIntStrategy"; }
};

} // namespace cvt_strategy
} // namespace ptxsim

#endif // PTXSIM_INSTRUCTIONS_CVT_CVT_FLOAT_TO_INT_H
