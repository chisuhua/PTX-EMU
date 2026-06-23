// cvt_int_to_int.h
// =============================================================================
// IntToIntStrategy (T2-6 Sub-task 4d — most complex)
//
// 处理 PTX CVT 指令的 !dst_is_float && !src_is_float 分支:
//   - 8/16/32/64 位整型 (signed 或 unsigned) ↔ 8/16/32/64 位整型
//   - 4×4 维度矩阵 + (signed, signed) ∈ {ss, su, us, uu}
//   - 含 .sat 饱和处理 (跨边界 clamp, signed↔unsigned 转换)
//   - 含 5 种舍入模式 (.rn/.rz/.rm/.rp/.rna + .rni/.rzi/.rmi/.rpi 别名)
//
// 复杂度: ~200 行 (Sub-task 4d: 最复杂的一个策略, 4×4×4 模板化)
// 来源: arithmetic_conversion.cpp 原 switch 中 case 1/2/4/8
//       !dst_is_float && !src_is_float 分支
// =============================================================================

#ifndef PTXSIM_INSTRUCTIONS_CVT_CVT_INT_TO_INT_H
#define PTXSIM_INSTRUCTIONS_CVT_CVT_INT_TO_INT_H

#include "ptxsim/instructions/cvt/cvt_strategy.h"

namespace ptxsim {
namespace cvt_strategy {

class IntToIntStrategy : public ConversionStrategy {
public:
    void convert(void *dst, void *src, const CvtContext &ctx) const override;
    const char *name() const override { return "IntToIntStrategy"; }
};

} // namespace cvt_strategy
} // namespace ptxsim

#endif // PTXSIM_INSTRUCTIONS_CVT_CVT_INT_TO_INT_H
