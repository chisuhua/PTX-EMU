#ifndef PTX_FLOAT_ARITH_CUH
#define PTX_FLOAT_ARITH_CUH

#include <cuda_runtime.h>
#include <cstdint>

// --- Device-side PTX wrappers (inline assembly) for float operations ---
__device__ __forceinline__ float ptx_add_f32(float a, float b) {
    float res;
    asm("add.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
    return res;
}

__device__ __forceinline__ float ptx_sub_f32(float a, float b) {
    float res;
    asm("sub.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
    return res;
}

__device__ __forceinline__ float ptx_mul_f32(float a, float b) {
    float res;
    asm("mul.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
    return res;
}

__device__ __forceinline__ float ptx_div_f32(float a, float b) {
    float res;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
    return res;
}

__device__ __forceinline__ float ptx_abs_f32(float a) {
    float res;
    asm("abs.f32 %0, %1;" : "=f"(res) : "f"(a));
    return res;
}

__device__ __forceinline__ float ptx_neg_f32(float a) {
    float res;
    asm("neg.f32 %0, %1;" : "=f"(res) : "f"(a));
    return res;
}

__device__ __forceinline__ float ptx_min_f32(float a, float b) {
    float res;
    asm("min.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
    return res;
}

__device__ __forceinline__ float ptx_max_f32(float a, float b) {
    float res;
    asm("max.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
    return res;
}

__device__ __forceinline__ float ptx_sqrt_f32(float a) {
    float res;
    asm("sqrt.rn.f32 %0, %1;" : "=f"(res) : "f"(a));
    return res;
}

__device__ __forceinline__ float ptx_rcp_f32(float a) {
    float res;
    asm("rcp.rn.f32 %0, %1;" : "=f"(res) : "f"(a));
    return res;
}


// --- Device-side PTX wrappers (inline assembly) for double operations ---
__device__ __forceinline__ double ptx_add_f64(double a, double b) {
    double res;
    asm("add.f64 %0, %1, %2;" : "=d"(res) : "d"(a), "d"(b));
    return res;
}

__device__ __forceinline__ double ptx_sub_f64(double a, double b) {
    double res;
    asm("sub.f64 %0, %1, %2;" : "=d"(res) : "d"(a), "d"(b));
    return res;
}

__device__ __forceinline__ double ptx_mul_f64(double a, double b) {
    double res;
    asm("mul.f64 %0, %1, %2;" : "=d"(res) : "d"(a), "d"(b));
    return res;
}

__device__ __forceinline__ double ptx_div_f64(double a, double b) {
    double res;
    asm("div.rn.f64 %0, %1, %2;" : "=d"(res) : "d"(a), "d"(b));
    return res;
}

__device__ __forceinline__ double ptx_abs_f64(double a) {
    double res;
    asm("abs.f64 %0, %1;" : "=d"(res) : "d"(a));
    return res;
}

__device__ __forceinline__ double ptx_neg_f64(double a) {
    double res;
    asm("neg.f64 %0, %1;" : "=d"(res) : "d"(a));
    return res;
}

__device__ __forceinline__ double ptx_min_f64(double a, double b) {
    double res;
    asm("min.f64 %0, %1, %2;" : "=d"(res) : "d"(a), "d"(b));
    return res;
}

__device__ __forceinline__ double ptx_max_f64(double a, double b) {
    double res;
    asm("max.f64 %0, %1, %2;" : "=d"(res) : "d"(a), "d"(b));
    return res;
}

__device__ __forceinline__ double ptx_sqrt_f64(double a) {
    double res;
    asm("sqrt.rn.f64 %0, %1;" : "=d"(res) : "d"(a));
    return res;
}

__device__ __forceinline__ double ptx_rcp_f64(double a) {
    double res;
    asm("rcp.rn.f64 %0, %1;" : "=d"(res) : "d"(a));
    return res;
}

// --- Host-side wrapper functions for float ---
void test_ptx_add_f32(float a, float b, float* result);
void test_ptx_sub_f32(float a, float b, float* result);
void test_ptx_mul_f32(float a, float b, float* result);
void test_ptx_div_f32(float a, float b, float* result);
void test_ptx_abs_f32(float a, float* result);
void test_ptx_neg_f32(float a, float* result);
void test_ptx_min_f32(float a, float b, float* result);
void test_ptx_max_f32(float a, float b, float* result);
void test_ptx_sqrt_f32(float a, float* result);
void test_ptx_rcp_f32(float a, float* result);


// --- Host-side wrapper functions for double ---
void test_ptx_add_f64(double a, double b, double* result);
void test_ptx_sub_f64(double a, double b, double* result);
void test_ptx_mul_f64(double a, double b, double* result);
void test_ptx_div_f64(double a, double b, double* result);
void test_ptx_abs_f64(double a, double* result);
void test_ptx_neg_f64(double a, double* result);
void test_ptx_min_f64(double a, double b, double* result);
void test_ptx_max_f64(double a, double b, double* result);
void test_ptx_sqrt_f64(double a, double* result);
void test_ptx_rcp_f64(double a, double* result);

#endif // PTX_FLOAT_ARITH_CUH