#ifndef PTX_CVT_ARITH_H
#define PTX_CVT_ARITH_H

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Integer to integer conversions
void test_ptx_cvt_s8_s16(int16_t a, int8_t *result);
void test_ptx_cvt_s8_s32(int32_t a, int8_t *result);
void test_ptx_cvt_s8_s64(int64_t a, int8_t *result);
void test_ptx_cvt_u8_u16(uint16_t a, uint8_t *result);
void test_ptx_cvt_u8_u32(uint32_t a, uint8_t *result);
void test_ptx_cvt_u8_u64(uint64_t a, uint8_t *result);
void test_ptx_cvt_s16_s8(int8_t a, int16_t *result);
void test_ptx_cvt_s16_s32(int32_t a, int16_t *result);
void test_ptx_cvt_s16_s64(int64_t a, int16_t *result);
void test_ptx_cvt_u16_u8(uint8_t a, uint16_t *result);
void test_ptx_cvt_u16_u32(uint32_t a, uint16_t *result);
void test_ptx_cvt_u16_u64(uint64_t a, uint16_t *result);
void test_ptx_cvt_s32_s8(int8_t a, int32_t *result);
void test_ptx_cvt_s32_s16(int16_t a, int32_t *result);
void test_ptx_cvt_s32_s64(int64_t a, int32_t *result);
void test_ptx_cvt_u32_u8(uint8_t a, uint32_t *result);
void test_ptx_cvt_u32_u16(uint16_t a, uint32_t *result);
void test_ptx_cvt_u32_u64(uint64_t a, uint32_t *result);
void test_ptx_cvt_s64_s8(int8_t a, int64_t *result);
void test_ptx_cvt_s64_s16(int16_t a, int64_t *result);
void test_ptx_cvt_s64_s32(int32_t a, int64_t *result);
void test_ptx_cvt_u64_u8(uint8_t a, uint64_t *result);
void test_ptx_cvt_u64_u16(uint16_t a, uint64_t *result);
void test_ptx_cvt_u64_u32(uint32_t a, uint64_t *result);

// Float to integer conversions
void test_ptx_cvt_s32_f32(float a, int32_t *result);
void test_ptx_cvt_u32_f32(float a, uint32_t *result);
void test_ptx_cvt_s64_f64(double a, int64_t *result);
void test_ptx_cvt_u64_f64(double a, uint64_t *result);
void test_ptx_cvt_s32_f64(double a, int32_t *result);
void test_ptx_cvt_u32_f64(double a, uint32_t *result);

// Rounding mode conversions (non-saturated)
void test_ptx_cvt_u8_f32_rni(float a, uint8_t *result);
void test_ptx_cvt_u8_f32_rzi(float a, uint8_t *result);
void test_ptx_cvt_u8_f32_rmi(float a, uint8_t *result);
void test_ptx_cvt_u8_f32_rpi(float a, uint8_t *result);
void test_ptx_cvt_u16_f32_rni(float a, uint16_t *result);
void test_ptx_cvt_u16_f32_rzi(float a, uint16_t *result);
void test_ptx_cvt_u16_f32_rmi(float a, uint16_t *result);
void test_ptx_cvt_u16_f32_rpi(float a, uint16_t *result);
void test_ptx_cvt_u32_f32_rni(float a, uint32_t *result);
void test_ptx_cvt_u32_f32_rzi(float a, uint32_t *result);
void test_ptx_cvt_u32_f32_rmi(float a, uint32_t *result);
void test_ptx_cvt_u32_f32_rpi(float a, uint32_t *result);

// Integer to float conversions
void test_ptx_cvt_f32_s32(int32_t a, float *result);
void test_ptx_cvt_f32_u32(uint32_t a, float *result);
void test_ptx_cvt_f64_s64(int64_t a, double *result);
void test_ptx_cvt_f64_u64(uint64_t a, double *result);
void test_ptx_cvt_f32_s64(int64_t a, float *result);
void test_ptx_cvt_f32_u64(uint64_t a, float *result);

// Float to float conversions
void test_ptx_cvt_f32_f64(double a, float *result);
void test_ptx_cvt_f64_f32(float a, double *result);

// Half precision (f16) to float (f32) conversions
void test_ptx_cvt_f32_f16(__half a, float *result);
void test_ptx_cvt_f16_f32(float a, __half *result);

// Saturation conversions
void test_ptx_cvt_satu8_f32(float a, uint8_t *result);
void test_ptx_cvt_satu16_f32(float a, uint16_t *result);
void test_ptx_cvt_satu32_f32(float a, uint32_t *result);

// Device function declarations
__device__ int8_t ptx_cvt_s8_s16(int16_t a);
__device__ int8_t ptx_cvt_s8_s32(int32_t a);
__device__ int8_t ptx_cvt_s8_s64(int64_t a);
__device__ uint8_t ptx_cvt_u8_u16(uint16_t a);
__device__ uint8_t ptx_cvt_u8_u32(uint32_t a);
__device__ uint8_t ptx_cvt_u8_u64(uint64_t a);
__device__ int16_t ptx_cvt_s16_s8(int8_t a);
__device__ int16_t ptx_cvt_s16_s32(int32_t a);
__device__ int16_t ptx_cvt_s16_s64(int64_t a);
__device__ uint16_t ptx_cvt_u16_u8(uint8_t a);
__device__ uint16_t ptx_cvt_u16_u32(uint32_t a);
__device__ uint16_t ptx_cvt_u16_u64(uint64_t a);
__device__ int32_t ptx_cvt_s32_s8(int8_t a);
__device__ int32_t ptx_cvt_s32_s16(int16_t a);
__device__ int32_t ptx_cvt_s32_s64(int64_t a);
__device__ uint32_t ptx_cvt_u32_u8(uint8_t a);
__device__ uint32_t ptx_cvt_u32_u16(uint16_t a);
__device__ uint32_t ptx_cvt_u32_u64(uint64_t a);
__device__ int64_t ptx_cvt_s64_s8(int8_t a);
__device__ int64_t ptx_cvt_s64_s16(int16_t a);
__device__ int64_t ptx_cvt_s64_s32(int32_t a);
__device__ uint64_t ptx_cvt_u64_u8(uint8_t a);
__device__ uint64_t ptx_cvt_u64_u16(uint16_t a);
__device__ uint64_t ptx_cvt_u64_u32(uint32_t a);

// Float to integer conversions
__device__ int32_t ptx_cvt_s32_f32(float a);
__device__ uint32_t ptx_cvt_u32_f32(float a);
__device__ int64_t ptx_cvt_s64_f64(double a);
__device__ uint64_t ptx_cvt_u64_f64(double a);
__device__ int32_t ptx_cvt_s32_f64(double a);
__device__ uint32_t ptx_cvt_u32_f64(double a);

// Rounding mode conversions (non-saturated)
__device__ uint8_t ptx_cvt_u8_f32_rni(float a);
__device__ uint8_t ptx_cvt_u8_f32_rzi(float a);
__device__ uint8_t ptx_cvt_u8_f32_rmi(float a);
__device__ uint8_t ptx_cvt_u8_f32_rpi(float a);
__device__ uint16_t ptx_cvt_u16_f32_rni(float a);
__device__ uint16_t ptx_cvt_u16_f32_rzi(float a);
__device__ uint16_t ptx_cvt_u16_f32_rmi(float a);
__device__ uint16_t ptx_cvt_u16_f32_rpi(float a);
__device__ uint32_t ptx_cvt_u32_f32_rni(float a);
__device__ uint32_t ptx_cvt_u32_f32_rzi(float a);
__device__ uint32_t ptx_cvt_u32_f32_rmi(float a);
__device__ uint32_t ptx_cvt_u32_f32_rpi(float a);

// Integer to float conversions
__device__ float ptx_cvt_f32_s32(int32_t a);
__device__ float ptx_cvt_f32_u32(uint32_t a);
__device__ double ptx_cvt_f64_s64(int64_t a);
__device__ double ptx_cvt_f64_u64(uint64_t a);
__device__ float ptx_cvt_f32_s64(int64_t a);
__device__ float ptx_cvt_f32_u64(uint64_t a);

// Float to float conversions
__device__ float ptx_cvt_f32_f64(double a);
__device__ double ptx_cvt_f64_f32(float a);

// Half precision (f16) to float (f32) conversions
__device__ float ptx_cvt_f32_f16(__half a);
__device__ __half ptx_cvt_f16_f32(float a);

// Saturation conversions
__device__ uint8_t ptx_cvt_satu8_f32(float a);
__device__ uint16_t ptx_cvt_satu16_f32(float a);
__device__ uint32_t ptx_cvt_satu32_f32(float a);

#endif // PTX_CVT_ARITH_H