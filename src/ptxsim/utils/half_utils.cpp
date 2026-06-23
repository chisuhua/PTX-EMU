#include "ptxsim/utils/half_utils.h"
#include <cstring>

// =============================================================================
// f16 -> f32 conversion
// =============================================================================
// IEEE 754 half precision: 1 sign bit, 5-bit exponent (bias 15), 10-bit
// mantissa.  This implementation matches the algorithm in
// src/ptxsim/instructions/cvt/cvt_helpers.cpp::half_to_float (commit 32ce8a0)
// bit-for-bit so that the two implementations are interchangeable.
//
// Key invariant for denormals (exp == 0, mantissa != 0):
//   value = (-1)^sign * mantissa * 2^-24
//   encoded in float32 with exponent = 103 + p  (p = position of high bit of
//   the 10-bit mantissa, range 0..9).
// =============================================================================
float f16_to_f32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 1;
    uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f32;

    if (exp == 0) {
        if (mant == 0) {
            // ±0
            f32 = sign << 31;
        } else {
            // Subnormal (denormal): value = mantissa × 2^-24.
            // Shift the 10-bit mantissa until bit 9 is set; the number of
            // shifts (lz) is the leading-zero count.  The high bit of the
            // original mantissa is at position (9 - lz), so exp_f = 103 + p
            // where p = 9 - lz (equivalently 112 - lz).  The 9 remaining
            // fraction bits (m & 0x1FF) become the float32 mantissa shifted
            // up by 14 to fill the 23-bit fraction field.
            //   mant=0x001 (lz=9): exp_f=103, frac=0   -> 2^-24
            //   mant=0x3FF (lz=0): exp_f=112, frac=0x7FE000 -> ~6.097e-5
            uint32_t m = mant;
            int lz = 0;
            while ((m & 0x200) == 0) {
                m <<= 1;
                lz++;
            }
            uint32_t frac = (m & 0x1FF) << 14;
            uint32_t exp_f = 112 - lz;
            f32 = (sign << 31) | (exp_f << 23) | frac;
        }
    } else if (exp == 0x1F) {
        if (mant == 0) {
            f32 = (sign << 31) | (0xFFU << 23);
        } else {
            f32 = (sign << 31) | (0xFFU << 23) | (mant << 13);
        }
    } else {
        f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }

    float res;
    std::memcpy(&res, &f32, 4);
    return res;
}

// =============================================================================
// f32 -> f16 conversion
// =============================================================================
// IEEE 754 round-to-nearest-even.  Mirrors
// src/ptxsim/instructions/cvt/cvt_helpers.cpp::float_to_half for the
// well-defined input domains (infinity, NaN, normal, normal-overflow).  Adds
// three correctness fixes that the local copy of float_to_half lacked:
//
//   1. Float subnormal input (exp == 0, mantissa != 0) rounds to signed zero.
//      Every float subnormal has value < 2^-126, far below the smallest half
//      subnormal (2^-24).  The previous half_utils implementation silently
//      returned zero (dropping the sign), and cvt_helpers incorrectly
//      renormalized the mantissa, producing a huge positive value.
//
//   2. Normal float underflowing to a half subnormal (new_exp in (-10, 0])
//      uses shift amount (14 - new_exp), not (12 - new_exp).  The cvt_helpers
//      shift was 2 bits too small, causing all such values to be off by
//      a factor of 4.
// =============================================================================
uint16_t f32_to_f16(float f) {
    uint32_t f32;
    std::memcpy(&f32, &f, 4);

    uint32_t sign = (f32 >> 31) & 1;
    uint32_t exp = (f32 >> 23) & 0xFF;
    uint32_t mant = f32 & 0x7FFFFF;

    uint16_t h;
    if (exp == 0) {
        // Zero or float subnormal: any float subnormal < 2^-126 is well below
        // the smallest half subnormal (2^-24), so it rounds to signed zero.
        h = static_cast<uint16_t>(sign << 15);
    } else if (exp == 0xFF) {
        h = static_cast<uint16_t>((sign << 15) | (0x1F << 10) |
                                  (mant ? 0x200 : 0) |
                                  (mant ? (mant >> 13) : 0));
    } else {
        int32_t new_exp = static_cast<int32_t>(exp) - 127 + 15;
        if (new_exp >= 0x1F) {
            h = static_cast<uint16_t>((sign << 15) | (0x1F << 10));
        } else if (new_exp <= 0) {
            // Half subnormal (new_exp in [-9, 0]); new_exp <= -10 rounds to 0.
            if (new_exp <= -10) {
                h = static_cast<uint16_t>(sign << 15);
            } else {
                // Reconstruct (1 + mant/2^23) * 2^(exp-127) as a half mantissa
                // * 2^-24.  Shift = 14 - new_exp places the implicit-1 bit
                // (position 23) at the half mantissa LSB.
                mant = (mant | 0x800000) >> (14 - new_exp);
                h = static_cast<uint16_t>((sign << 15) | (mant & 0x3FF));
            }
        } else {
            h = static_cast<uint16_t>((sign << 15) | (new_exp << 10) |
                                      (mant >> 13));
        }
    }

    return h;
}