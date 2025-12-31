#ifndef PTXSIM_HALF_UTILS_H
#define PTXSIM_HALF_UTILS_H

#include <cstdint>

// Helper function to convert f16 to f32 (simplified)
float f16_to_f32(uint16_t h);

// Helper function to convert f32 to f16 (simplified)
uint16_t f32_to_f16(float f);

#endif // PTXSIM_HALF_UTILS_H