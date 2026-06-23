// cvt_helpers.cpp
// =============================================================================
// 实现说明：本文件第一版直接复制 arithmetic_conversion.cpp 的 4 个 helper
// （line 11-139），namespace 改为 ptxsim::cvt_helpers。
// Step 7-8 将复用 half_utils.h 替换重复的 half_to_float/float_to_half。
// =============================================================================

#include "ptxsim/instructions/cvt/cvt_helpers.h"
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

// 自定义half到float的转换函数
float half_to_float(uint16_t h) {
    uint32_t sign = ((h >> 15) & 0x1);
    uint32_t exp = ((h >> 10) & 0x1f);
    uint32_t mantissa = (h & 0x3ff);
    uint32_t f;

    if (exp == 0) {
        if (mantissa == 0) {
            // ±0
            f = sign << 31;
        } else {
            // Subnormal (denormal) numbers: value = mantissa × 2^-24
            // Shift the 10-bit mantissa until bit 9 is set; the number of
            // shifts (lz) is the leading-zero count in the 10-bit value.
            // The high bit of the original mantissa is at position (9 - lz),
            // so exp_f = 103 + (9 - lz) = 112 - lz, and the 9 remaining
            // fraction bits (m & 0x1FF) become the float32 mantissa shifted
            // up by 14 to fill the 23-bit fraction field.
            // For mantissa=1 (lz=9): exp_f=103, frac=0 → 0x33800000 = 2^-24
            // For mantissa=0x3FF (lz=0): exp_f=112, frac=0x7FE000
            uint32_t m = mantissa;
            int lz = 0;
            while ((m & 0x200) == 0) {
                m <<= 1;
                lz++;
            }
            uint32_t frac = (m & 0x1FF) << 14;
            uint32_t exp_f = 112 - lz;
            f = (sign << 31) | (exp_f << 23) | frac;
        }
    } else if (exp == 31) {
        if (mantissa == 0) {
            // ±infinity
            f = (sign << 31) | (0xFF << 23);
        } else {
            // NaN
            f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        }
    } else {
        // Normalized number
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }

    return *reinterpret_cast<float *>(&f);
}

// Check if a float value should saturate to UINT32_MAX when converting to
// uint32. Handles the precision issue where float32 cannot exactly represent
// values in [4294967295.0, 4294967296.0), causing values like 4294967295.4f to
// become 4294967296.0f in float32 representation.
bool should_saturate_uint32(float temp, float sat_high) {
    return temp >= 4294967295.0f && temp <= sat_high;
}

// 自定义float到half的转换函数
uint16_t float_to_half(float f) {
    uint32_t bits = *reinterpret_cast<uint32_t *>(&f);
    uint16_t sign = (bits >> 16) & 0x8000;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;

    uint16_t result;

    if (exp == 0) {
        // Zero or subnormal
        if (mantissa == 0) {
            result = sign; // +0 or -0
        } else {
            // Float subnormal -> half might be normal or subnormal
            // Need to normalize the mantissa and calculate the new exponent
            int shift = 0;
            while ((mantissa & 0x800000) == 0) {
                mantissa <<= 1;
                shift++;
            }
            exp = 127 - shift;    // original exp was 0, so real exp is -126
            exp = exp - 127 + 15; // Convert to half exponent
            if (exp <= 0) {
                // Result is subnormal in half
                mantissa = (mantissa & 0x7FFFFF) >> 13;
                if (exp == 0) {
                    // Check if we need to shift right based on exponent
                    // difference
                    mantissa |= 0x400; // Add the implicit bit
                    mantissa >>= 1;
                } else {
                    mantissa >>= (1 - exp);
                }
                result = sign | mantissa;
            } else {
                // Normal half number
                mantissa >>= 13;
                result = sign | (exp << 10) | (mantissa & 0x3FF);
            }
        }
    } else if (exp == 0xFF) {
        // infinity or NaN
        result = sign | (0x1F << 10) | (mantissa != 0 ? 0x200 : 0);
    } else {
        // Normal float number
        exp = exp - 127 + 15; // Convert to half exponent
        if (exp >= 0x1F) {
            // Overflow - infinity
            result = sign | (0x1F << 10);
        } else if (exp <= 0) {
            // Underflow - subnormal or zero
            if (exp <= -10) {
                // Rounds to zero
                result = sign;
            } else {
                // Convert to subnormal
                mantissa = (mantissa | 0x800000) >>
                           (12 - exp); // Add implicit bit and shift
                result = sign | (mantissa >> 13);
            }
        } else {
            // Normal half number
            result = sign | (exp << 10) | (mantissa >> 13);
        }
    }

    return result;
}

} // namespace cvt_helpers
} // namespace ptxsim