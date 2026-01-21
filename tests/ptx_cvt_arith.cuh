#ifndef PTX_CVT_ARITH_CUH
#define PTX_CVT_ARITH_CUH

#include <cuda_runtime.h>
#include <cstdint>

// --- Device-side PTX wrappers (inline assembly) for CVT operations ---

// Integer to integer conversions
__device__ __forceinline__ int8_t ptx_cvt_s8_s16(int16_t a) {
    int32_t temp;
    asm("cvt.s8.s16 %0, %1;" : "=r"(temp) : "h"(a));
    return static_cast<int8_t>(static_cast<uint8_t>(temp)); // 强制低8位
}

__device__ __forceinline__ int8_t ptx_cvt_s8_s32(int32_t a) {
    int32_t temp;
    asm("cvt.s8.s32 %0, %1;" : "=r"(temp) : "r"(a));
    return static_cast<int8_t>(static_cast<uint8_t>(temp)); // 强制低8位
}

__device__ __forceinline__ int8_t ptx_cvt_s8_s64(int64_t a) {
    int32_t temp;
    asm("cvt.s8.s64 %0, %1;" : "=r"(temp) : "l"(a));
    return static_cast<int8_t>(static_cast<uint8_t>(temp)); // 强制低8位
}

__device__ __forceinline__ uint8_t ptx_cvt_u8_u16(uint16_t a) {
    uint32_t temp;
    asm("cvt.u8.u16 %0, %1;" : "=r"(temp) : "h"(a));
    return static_cast<uint8_t>(static_cast<uint8_t>(temp)); // 强制低8位
}

__device__ __forceinline__ uint8_t ptx_cvt_u8_u32(uint32_t a) {
    uint32_t temp;
    asm("cvt.u8.u32 %0, %1;" : "=r"(temp) : "r"(a));
    return static_cast<uint8_t>(static_cast<uint8_t>(temp)); // 强制低8位
}

__device__ __forceinline__ uint8_t ptx_cvt_u8_u64(uint64_t a) {
    uint32_t temp;
    asm("cvt.u8.u64 %0, %1;" : "=r"(temp) : "l"(a));
    return static_cast<uint8_t>(static_cast<uint8_t>(temp)); // 强制低8位
}

__device__ __forceinline__ int16_t ptx_cvt_s16_s8(int8_t a) {
    int32_t temp;
    asm("cvt.s16.s8 %0, %1;" : "=r"(temp) : "r"((int32_t)a));
    return static_cast<int16_t>(static_cast<uint16_t>(temp)); // 强制低16位
}

__device__ __forceinline__ int16_t ptx_cvt_s16_s32(int32_t a) {
    int16_t res;
    asm("cvt.s16.s32 %0, %1;" : "=h"(res) : "r"(a));
    return res;
}

__device__ __forceinline__ int16_t ptx_cvt_s16_s64(int64_t a) {
    int16_t res;
    asm("cvt.s16.s64 %0, %1;" : "=h"(res) : "l"(a));
    return res;
}

__device__ __forceinline__ uint16_t ptx_cvt_u16_u8(uint8_t a) {
    uint32_t temp;
    asm("cvt.u16.u8 %0, %1;" : "=r"(temp) : "r"((uint32_t)a));
    return static_cast<uint16_t>(static_cast<uint16_t>(temp)); // 强制低16位
}

__device__ __forceinline__ uint16_t ptx_cvt_u16_u32(uint32_t a) {
    uint16_t res;
    asm("cvt.u16.u32 %0, %1;" : "=h"(res) : "r"(a));
    return res;
}

__device__ __forceinline__ uint16_t ptx_cvt_u16_u64(uint64_t a) {
    uint16_t res;
    asm("cvt.u16.u64 %0, %1;" : "=h"(res) : "l"(a));
    return res;
}

__device__ __forceinline__ int32_t ptx_cvt_s32_s8(int8_t a) {
    int32_t temp;
    asm("cvt.s32.s8 %0, %1;" : "=r"(temp) : "r"((int32_t)a));
    return temp; // 32位目标，无需特殊处理
}

__device__ __forceinline__ int32_t ptx_cvt_s32_s16(int16_t a) {
    int32_t temp;
    asm("cvt.s32.s16 %0, %1;" : "=r"(temp) : "h"(a));
    return temp; // 32位目标，无需特殊处理
}

__device__ __forceinline__ int32_t ptx_cvt_s32_s64(int64_t a) {
    int32_t temp;
    asm("cvt.s32.s64 %0, %1;" : "=r"(temp) : "l"(a));
    return temp; // 32位目标，无需特殊处理
}

__device__ __forceinline__ uint32_t ptx_cvt_u32_u8(uint8_t a) {
    uint32_t temp;
    asm("cvt.u32.u8 %0, %1;" : "=r"(temp) : "r"((uint32_t)a));
    return temp; // 32位目标，无需特殊处理
}

__device__ __forceinline__ uint32_t ptx_cvt_u32_u16(uint16_t a) {
    uint32_t temp;
    asm("cvt.u32.u16 %0, %1;" : "=r"(temp) : "h"(a));
    return temp; // 32位目标，无需特殊处理
}

__device__ __forceinline__ uint32_t ptx_cvt_u32_u64(uint64_t a) {
    uint32_t temp;
    asm("cvt.u32.u64 %0, %1;" : "=r"(temp) : "l"(a));
    return temp; // 32位目标，无需特殊处理
}

__device__ __forceinline__ int64_t ptx_cvt_s64_s8(int8_t a) {
    int64_t res;
    int64_t temp;
    asm("cvt.s64.s8 %0, %1;" : "=l"(temp) : "r"((int32_t)a));
    res = (int64_t)(temp);
    return res;
}

__device__ __forceinline__ int64_t ptx_cvt_s64_s16(int16_t a) {
    int64_t res;
    int64_t temp;
    asm("cvt.s64.s16 %0, %1;" : "=l"(temp) : "h"(a));
    res = static_cast<int64_t>(temp);
    return res;
}

__device__ __forceinline__ int64_t ptx_cvt_s64_s32(int32_t a) {
    int64_t res;
    int64_t temp;
    asm("cvt.s64.s32 %0, %1;" : "=l"(temp) : "r"(a));
    res = static_cast<int64_t>(temp);
    return res;
}

__device__ __forceinline__ uint64_t ptx_cvt_u64_u8(uint8_t a) {
    uint64_t res;
    uint64_t temp;
    asm("cvt.u64.u8 %0, %1;" : "=l"(temp) : "r"((uint32_t)a));
    res = (uint64_t)(temp);
    return res;
}

__device__ __forceinline__ uint64_t ptx_cvt_u64_u16(uint16_t a) {
    uint64_t res;
    uint64_t temp;
    asm("cvt.u64.u16 %0, %1;" : "=l"(temp) : "h"(a));
    res = static_cast<uint64_t>(temp);
    return res;
}

__device__ __forceinline__ uint64_t ptx_cvt_u64_u32(uint32_t a) {
    uint64_t res;
    uint64_t temp;
    asm("cvt.u64.u32 %0, %1;" : "=l"(temp) : "r"(a));
    res = static_cast<uint64_t>(temp);
    return res;
}

// Float to integer conversions
__device__ __forceinline__ int32_t ptx_cvt_s32_f32(float a) {
    int32_t res;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(res) : "f"(a));
    return res;
}

__device__ __forceinline__ uint32_t ptx_cvt_u32_f32(float a) {
    uint32_t res;
    asm("cvt.rni.u32.f32 %0, %1;" : "=r"(res) : "f"(a));
    return res;
}

__device__ __forceinline__ int64_t ptx_cvt_s64_f64(double a) {
    int64_t res;
    asm("cvt.rni.s64.f64 %0, %1;" : "=l"(res) : "d"(a));
    return res;
}

__device__ __forceinline__ uint64_t ptx_cvt_u64_f64(double a) {
    uint64_t res;
    asm("cvt.rni.u64.f64 %0, %1;" : "=l"(res) : "d"(a));
    return res;
}

__device__ __forceinline__ int32_t ptx_cvt_s32_f64(double a) {
    int32_t res;
    asm("cvt.rni.s32.f64 %0, %1;" : "=r"(res) : "d"(a));
    return res;
}

__device__ __forceinline__ uint32_t ptx_cvt_u32_f64(double a) {
    uint32_t res;
    asm("cvt.rni.u32.f64 %0, %1;" : "=r"(res) : "d"(a));
    return res;
}

// Integer to float conversions
__device__ __forceinline__ float ptx_cvt_f32_s32(int32_t a) {
    float res;
    asm("cvt.rn.f32.s32 %0, %1;" : "=f"(res) : "r"(a));
    return res;
}

__device__ __forceinline__ float ptx_cvt_f32_u32(uint32_t a) {
    float res;
    asm("cvt.rn.f32.u32 %0, %1;" : "=f"(res) : "r"(a));
    return res;
}

__device__ __forceinline__ double ptx_cvt_f64_s64(int64_t a) {
    double res;
    asm("cvt.rn.f64.s64 %0, %1;" : "=d"(res) : "l"(a));
    return res;
}

__device__ __forceinline__ double ptx_cvt_f64_u64(uint64_t a) {
    double res;
    asm("cvt.rn.f64.u64 %0, %1;" : "=d"(res) : "l"(a));
    return res;
}

__device__ __forceinline__ float ptx_cvt_f32_s64(int64_t a) {
    float res;
    asm("cvt.rn.f32.s64 %0, %1;" : "=f"(res) : "l"(a));
    return res;
}

__device__ __forceinline__ float ptx_cvt_f32_u64(uint64_t a) {
    float res;
    asm("cvt.rn.f32.u64 %0, %1;" : "=f"(res) : "l"(a));
    return res;
}

// Float to float conversions
__device__ __forceinline__ float ptx_cvt_f32_f64(double a) {
    float res;
    // 从f64到f32的转换需要舍入模式，但从f32到f64的转换不需要
    asm("cvt.rn.f32.f64 %0, %1;" : "=f"(res) : "d"(a));
    return res;
}

__device__ __forceinline__ double ptx_cvt_f64_f32(float a) {
    double res;
    // 从f32到f64的转换是精确的，不需要舍入模式
    asm("cvt.f64.f32 %0, %1;" : "=d"(res) : "f"(a));
    return res;
}

// Saturation conversions
__device__ __forceinline__ uint8_t ptx_cvt_satu8_f32(float a) {
    uint8_t res;
    uint32_t temp_res;
    asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(temp_res) : "f"(a));
    res = (uint8_t)(temp_res);
    return res;
}

__device__ __forceinline__ uint16_t ptx_cvt_satu16_f32(float a) {
    uint16_t res;
    asm("cvt.rni.sat.u16.f32 %0, %1;" : "=h"(res) : "f"(a));
    return res;
}

__device__ __forceinline__ uint32_t ptx_cvt_satu32_f32(float a) {
    uint32_t res;
    asm("cvt.rni.sat.u32.f32 %0, %1;" : "=r"(res) : "f"(a));
    return res;
}

// --- Host-side wrapper functions ---

void test_ptx_cvt_s8_s16(int16_t a, int8_t* result);
void test_ptx_cvt_s8_s32(int32_t a, int8_t* result);
void test_ptx_cvt_s8_s64(int64_t a, int8_t* result);
void test_ptx_cvt_u8_u16(uint16_t a, uint8_t* result);
void test_ptx_cvt_u8_u32(uint32_t a, uint8_t* result);
void test_ptx_cvt_u8_u64(uint64_t a, uint8_t* result);
void test_ptx_cvt_s16_s8(int8_t a, int16_t* result);
void test_ptx_cvt_s16_s32(int32_t a, int16_t* result);
void test_ptx_cvt_s16_s64(int64_t a, int16_t* result);
void test_ptx_cvt_u16_u8(uint8_t a, uint16_t* result);
void test_ptx_cvt_u16_u32(uint32_t a, uint16_t* result);
void test_ptx_cvt_u16_u64(uint64_t a, uint16_t* result);
void test_ptx_cvt_s32_s8(int8_t a, int32_t* result);
void test_ptx_cvt_s32_s16(int16_t a, int32_t* result);
void test_ptx_cvt_s32_s64(int64_t a, int32_t* result);
void test_ptx_cvt_u32_u8(uint8_t a, uint32_t* result);
void test_ptx_cvt_u32_u16(uint16_t a, uint32_t* result);
void test_ptx_cvt_u32_u64(uint64_t a, uint32_t* result);
void test_ptx_cvt_s64_s8(int8_t a, int64_t* result);
void test_ptx_cvt_s64_s16(int16_t a, int64_t* result);
void test_ptx_cvt_s64_s32(int32_t a, int64_t* result);
void test_ptx_cvt_u64_u8(uint8_t a, uint64_t* result);
void test_ptx_cvt_u64_u16(uint16_t a, uint64_t* result);
void test_ptx_cvt_u64_u32(uint32_t a, uint64_t* result);

void test_ptx_cvt_s32_f32(float a, int32_t* result);
void test_ptx_cvt_u32_f32(float a, uint32_t* result);
void test_ptx_cvt_s64_f64(double a, int64_t* result);
void test_ptx_cvt_u64_f64(double a, uint64_t* result);
void test_ptx_cvt_s32_f64(double a, int32_t* result);
void test_ptx_cvt_u32_f64(double a, uint32_t* result);

void test_ptx_cvt_f32_s32(int32_t a, float* result);
void test_ptx_cvt_f32_u32(uint32_t a, float* result);
void test_ptx_cvt_f64_s64(int64_t a, double* result);
void test_ptx_cvt_f64_u64(uint64_t a, double* result);
void test_ptx_cvt_f32_s64(int64_t a, float* result);
void test_ptx_cvt_f32_u64(uint64_t a, float* result);

void test_ptx_cvt_f32_f64(double a, float* result);
void test_ptx_cvt_f64_f32(float a, double* result);

void test_ptx_cvt_satu8_f32(float a, uint8_t* result);
void test_ptx_cvt_satu16_f32(float a, uint16_t* result);
void test_ptx_cvt_satu32_f32(float a, uint32_t* result);

#endif // PTX_CVT_ARITH_CUH