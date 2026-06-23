// cvt_helpers.cpp
// =============================================================================
// CVT 指令共享 helpers。
// half_to_float / float_to_half 委托给 half_utils.h（f16_to_f32 / f32_to_f16），
// 避免重复实现；bit-perfect 一致性由 tests/unit/utils/test_half_utils_consistency
// 验证（65536 case 全过）。
// round_half_to_even / should_saturate_uint32 保留本地实现（half_utils 未涵盖）。
// =============================================================================

#include "ptxsim/instructions/cvt/cvt_helpers.h"
#include "ptxsim/utils/half_utils.h"
#include <cmath>

namespace ptxsim {
namespace cvt_helpers {

// 银行家舍入法（Round half to even）- 用于 RNI 修饰符
float round_half_to_even(float x) {
    float rounded = std::round(x);
    float diff = std::abs(x - rounded);

    // 如果正好是 0.5，使用 ties to even
    if (diff == 0.5f) {
        // 如果结果是奇数，调整到偶数
        if (std::fmod(std::abs(rounded), 2.0f) == 1.0f) {
            return rounded - (x > 0 ? 1.0f : -1.0f);
        }
    }
    return rounded;
}

// half → float 转换 - 委托给 half_utils.h::f16_to_f32（commit fbb7a29 已修 denormal）
float half_to_float(uint16_t h) { return ::f16_to_f32(h); }

// Check if a float value should saturate to UINT32_MAX when converting to
// uint32. Handles the precision issue where float32 cannot exactly represent
// values in [4294967295.0, 4294967296.0), causing values like 4294967295.4f to
// become 4294967296.0f in float32 representation.
bool should_saturate_uint32(float temp, float sat_high) {
    return temp >= 4294967295.0f && temp <= sat_high;
}

// float → half 转换 - 委托给 half_utils.h::f32_to_f16
uint16_t float_to_half(float f) { return ::f32_to_f16(f); }

} // namespace cvt_helpers
} // namespace ptxsim
