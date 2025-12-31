#include "ptxsim/utils/half_utils.h"
#include <cstring>

// Helper function to convert f16 to f32 (simplified)
float f16_to_f32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 1;
    uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f32;

    if (exp == 0x1F) {
        // Infinity or NaN
        f32 = (sign << 31) | (0xFFU << 23) | (mant << 13);
    } else {
        f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }

    float res;
    std::memcpy(&res, &f32, 4);
    return res;
}

// Helper function to convert f32 to f16 (simplified)
uint16_t f32_to_f16(float f) {
    uint32_t f32;
    std::memcpy(&f32, &f, 4);

    uint32_t sign = (f32 >> 31) & 1;
    uint32_t exp = (f32 >> 23) & 0xFF;
    uint32_t mant = f32 & 0x7FFFFF;

    uint16_t h;
    if (exp == 0xFF) {
        // Infinity or NaN
        h = (sign << 15) | (0x1F << 10) | (mant ? 0x200 : 0) | (mant >> 13);
    } else {
        int32_t new_exp = static_cast<int32_t>(exp) - 112;
        if (new_exp >= 31) {
            // Overflow
            h = (sign << 15) | (0x1F << 10); // Infinity
        } else if (new_exp <= 0) {
            // Underflow
            h = (sign << 15); // Zero or subnormal
        } else {
            h = (sign << 15) | (new_exp << 10) | (mant >> 13);
        }
    }

    return h;
}