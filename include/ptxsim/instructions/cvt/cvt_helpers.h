// cvt_helpers.h
// =============================================================================
// CVT 指令共享 helpers（从 arithmetic_conversion.cpp 抽离）
// 抽离目的：消除 1288 行单文件 god method + 与 half_utils.h 复用
// =============================================================================

#ifndef PTXSIM_INSTRUCTIONS_CVT_CVT_HELPERS_H
#define PTXSIM_INSTRUCTIONS_CVT_CVT_HELPERS_H

#include <cstdint>

namespace ptxsim {
namespace cvt_helpers {

// 银行家舍入（用于 .rni 修饰符）
float round_half_to_even(float x);

// f16 → f32 位解析（与 half_utils.h::f16_to_f32 等价）
float half_to_float(uint16_t h);

// f32 → f16 位解析（与 half_utils.h::f32_to_f16 等价）
uint16_t float_to_half(float f);

// 饱和边界检测（用于 f32→u32 转换的 .sat 修饰符）
bool should_saturate_uint32(float temp, float sat_high);

} // namespace cvt_helpers
} // namespace ptxsim

#endif // PTXSIM_INSTRUCTIONS_CVT_CVT_HELPERS_H