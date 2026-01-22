#include "ptx_cvt_arith.h"
#include <stdexcept>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error("CUDA error: "/* + std::string(cudaGetErrorString(err))*/); \
    } \
} while(0)

// Kernels for integer to integer conversions
__global__ void test_cvt_s8_s16_kernel(int16_t a, int8_t* result) { 
    int8_t temp = ptx_cvt_s8_s16(a);
    *result = temp;
}

void test_ptx_cvt_s8_s16(int16_t a, int8_t* result) {
    int8_t *d_result;
    cudaMalloc(&d_result, sizeof(int8_t));
    test_cvt_s8_s16_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int8_t ptx_cvt_s8_s16(int16_t a) {
    short result;
    asm("cvt.s8.s16.sat %0, %1;" : "=h"(result) : "h"(a));
    return (int8_t)result;
}

__global__ void test_cvt_s8_s32_kernel(int32_t a, int8_t* result) { 
    int8_t temp = ptx_cvt_s8_s32(a);
    *result = temp;
}

void test_ptx_cvt_s8_s32(int32_t a, int8_t* result) {
    int8_t *d_result;
    cudaMalloc(&d_result, sizeof(int8_t));
    test_cvt_s8_s32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int8_t ptx_cvt_s8_s32(int32_t a) {
    short result;
    asm("cvt.s8.s32.sat %0, %1;" : "=h"(result) : "r"(a));
    return result;
}

__global__ void test_cvt_s8_s64_kernel(int64_t a, int8_t* result) { 
    int8_t temp = ptx_cvt_s8_s64(a);
    *result = temp;
}

void test_ptx_cvt_s8_s64(int64_t a, int8_t* result) {
    int8_t *d_result;
    cudaMalloc(&d_result, sizeof(int8_t));
    test_cvt_s8_s64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int8_t ptx_cvt_s8_s64(int64_t a) {
    short result;
    asm("cvt.s8.s64.sat %0, %1;" : "=h"(result) : "l"(a));
    return (int8_t)result;
}

__global__ void test_cvt_u8_u16_kernel(uint16_t a, uint8_t* result) { 
    uint8_t temp = ptx_cvt_u8_u16(a);
    *result = temp;
}

void test_ptx_cvt_u8_u16(uint16_t a, uint8_t* result) {
    uint8_t *d_result;
    cudaMalloc(&d_result, sizeof(uint8_t));
    test_cvt_u8_u16_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint8_t ptx_cvt_u8_u16(uint16_t a) {
    uint16_t result;
    asm("cvt.u8.u16.sat %0, %1;" : "=h"(result) : "h"(a));
    return (uint8_t)result;
}

__global__ void test_cvt_u8_u32_kernel(uint32_t a, uint8_t* result) { 
    uint8_t temp = ptx_cvt_u8_u32(a);
    *result = temp;
}

void test_ptx_cvt_u8_u32(uint32_t a, uint8_t* result) {
    uint8_t *d_result;
    cudaMalloc(&d_result, sizeof(uint8_t));
    test_cvt_u8_u32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint8_t ptx_cvt_u8_u32(uint32_t a) {
    uint16_t result;
    asm("cvt.u8.u32.sat %0, %1;" : "=h"(result) : "r"(a));
    return (uint8_t)result;
}

__global__ void test_cvt_u8_u64_kernel(uint64_t a, uint8_t* result) { 
    uint8_t temp = ptx_cvt_u8_u64(a);
    *result = temp;
}

void test_ptx_cvt_u8_u64(uint64_t a, uint8_t* result) {
    uint8_t *d_result;
    cudaMalloc(&d_result, sizeof(uint8_t));
    test_cvt_u8_u64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint8_t ptx_cvt_u8_u64(uint64_t a) {
    uint16_t result;
    asm("cvt.u8.u64.sat %0, %1;" : "=h"(result) : "l"(a));
    return (uint8_t)result;
}

__global__ void test_cvt_s16_s8_kernel(int8_t a, int16_t* result) { 
    int16_t temp = ptx_cvt_s16_s8(a);
    *result = temp;
}

void test_ptx_cvt_s16_s8(int8_t a, int16_t* result) {
    int16_t *d_result;
    cudaMalloc(&d_result, sizeof(int16_t));
    test_cvt_s16_s8_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int16_t ptx_cvt_s16_s8(int8_t a) {
    int16_t a_val = a;
    int16_t result;
    asm("cvt.s16.s8 %0, %1;" : "=h"(result) : "h"(a_val));
    return result;
}

__global__ void test_cvt_s16_s32_kernel(int32_t a, int16_t* result) { 
    int16_t temp = ptx_cvt_s16_s32(a);
    *result = temp;
}

void test_ptx_cvt_s16_s32(int32_t a, int16_t* result) {
    int16_t *d_result;
    cudaMalloc(&d_result, sizeof(int16_t));
    test_cvt_s16_s32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int16_t ptx_cvt_s16_s32(int32_t a) {
    int16_t result;
    asm("cvt.s16.s32 %0, %1;" : "=h"(result) : "r"(a));
    return result;
}

__global__ void test_cvt_s16_s64_kernel(int64_t a, int16_t* result) { 
    int16_t temp = ptx_cvt_s16_s64(a);
    *result = temp;
}

void test_ptx_cvt_s16_s64(int64_t a, int16_t* result) {
    int16_t *d_result;
    cudaMalloc(&d_result, sizeof(int16_t));
    test_cvt_s16_s64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int16_t ptx_cvt_s16_s64(int64_t a) {
    int16_t result;
    asm("cvt.s16.s64 %0, %1;" : "=h"(result) : "l"(a));
    return result;
}

__global__ void test_cvt_u16_u8_kernel(uint8_t a, uint16_t* result) { 
    uint16_t temp = ptx_cvt_u16_u8(a);
    *result = temp;
}

void test_ptx_cvt_u16_u8(uint8_t a, uint16_t* result) {
    uint16_t *d_result;
    cudaMalloc(&d_result, sizeof(uint16_t));
    test_cvt_u16_u8_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint16_t ptx_cvt_u16_u8(uint8_t a) {
    uint16_t result;
    uint16_t a_val = a;
    asm("cvt.u16.u8 %0, %1;" : "=h"(result) : "h"(a_val));
    return result;
}

__global__ void test_cvt_u16_u32_kernel(uint32_t a, uint16_t* result) { 
    uint16_t temp = ptx_cvt_u16_u32(a);
    *result = temp;
}

void test_ptx_cvt_u16_u32(uint32_t a, uint16_t* result) {
    uint16_t *d_result;
    cudaMalloc(&d_result, sizeof(uint16_t));
    test_cvt_u16_u32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint16_t ptx_cvt_u16_u32(uint32_t a) {
    uint16_t result;
    asm("cvt.u16.u32 %0, %1;" : "=h"(result) : "r"(a));
    return result;
}

__global__ void test_cvt_u16_u64_kernel(uint64_t a, uint16_t* result) { 
    uint16_t temp = ptx_cvt_u16_u64(a);
    *result = temp;
}

void test_ptx_cvt_u16_u64(uint64_t a, uint16_t* result) {
    uint16_t *d_result;
    cudaMalloc(&d_result, sizeof(uint16_t));
    test_cvt_u16_u64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint16_t ptx_cvt_u16_u64(uint64_t a) {
    uint16_t result;
    asm("cvt.u16.u64 %0, %1;" : "=h"(result) : "l"(a));
    return result;
}

__global__ void test_cvt_s32_s8_kernel(int8_t a, int32_t* result) { 
    int32_t temp = ptx_cvt_s32_s8(a);
    *result = temp;
}

void test_ptx_cvt_s32_s8(int8_t a, int32_t* result) {
    int32_t *d_result;
    cudaMalloc(&d_result, sizeof(int32_t));
    test_cvt_s32_s8_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int32_t ptx_cvt_s32_s8(int8_t a) {
    int32_t result;
    int16_t a_val = a;
    asm("cvt.s32.s8 %0, %1;" : "=r"(result) : "h"(a_val));
    return result;
}

__global__ void test_cvt_s32_s16_kernel(int16_t a, int32_t* result) { 
    int32_t temp = ptx_cvt_s32_s16(a);
    *result = temp;
}

void test_ptx_cvt_s32_s16(int16_t a, int32_t* result) {
    int32_t *d_result;
    cudaMalloc(&d_result, sizeof(int32_t));
    test_cvt_s32_s16_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int32_t ptx_cvt_s32_s16(int16_t a) {
    int32_t result;
    asm("cvt.s32.s16 %0, %1;" : "=r"(result) : "h"(a));
    return result;
}

__global__ void test_cvt_s32_s64_kernel(int64_t a, int32_t* result) { 
    int32_t temp = ptx_cvt_s32_s64(a);
    *result = temp;
}

void test_ptx_cvt_s32_s64(int64_t a, int32_t* result) {
    int32_t *d_result;
    cudaMalloc(&d_result, sizeof(int32_t));
    test_cvt_s32_s64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int32_t ptx_cvt_s32_s64(int64_t a) {
    int32_t result;
    asm("cvt.s32.s64 %0, %1;" : "=r"(result) : "l"(a));
    return result;
}

__global__ void test_cvt_u32_u8_kernel(uint8_t a, uint32_t* result) { 
    uint32_t temp = ptx_cvt_u32_u8(a);
    *result = temp;
}

void test_ptx_cvt_u32_u8(uint8_t a, uint32_t* result) {
    uint32_t *d_result;
    cudaMalloc(&d_result, sizeof(uint32_t));
    test_cvt_u32_u8_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint32_t ptx_cvt_u32_u8(uint8_t a) {
    uint32_t result;
    uint16_t a_val = a;
    asm("cvt.u32.u8 %0, %1;" : "=r"(result) : "h"(a_val));
    return result;
}

__global__ void test_cvt_u32_u16_kernel(uint16_t a, uint32_t* result) { 
    uint32_t temp = ptx_cvt_u32_u16(a);
    *result = temp;
}

void test_ptx_cvt_u32_u16(uint16_t a, uint32_t* result) {
    uint32_t *d_result;
    cudaMalloc(&d_result, sizeof(uint32_t));
    test_cvt_u32_u16_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint32_t ptx_cvt_u32_u16(uint16_t a) {
    uint32_t result;
    asm("cvt.u32.u16 %0, %1;" : "=r"(result) : "h"(a));
    return result;
}

__global__ void test_cvt_u32_u64_kernel(uint64_t a, uint32_t* result) { 
    uint32_t temp = ptx_cvt_u32_u64(a);
    *result = temp;
}

void test_ptx_cvt_u32_u64(uint64_t a, uint32_t* result) {
    uint32_t *d_result;
    cudaMalloc(&d_result, sizeof(uint32_t));
    test_cvt_u32_u64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint32_t ptx_cvt_u32_u64(uint64_t a) {
    uint32_t result;
    asm("cvt.u32.u64 %0, %1;" : "=r"(result) : "l"(a));
    return result;
}

__global__ void test_cvt_s64_s8_kernel(int8_t a, int64_t* result) { 
    int64_t temp = ptx_cvt_s64_s8(a);
    *result = temp;
}

void test_ptx_cvt_s64_s8(int8_t a, int64_t* result) {
    int64_t *d_result;
    cudaMalloc(&d_result, sizeof(int64_t));
    test_cvt_s64_s8_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int64_t ptx_cvt_s64_s8(int8_t a) {
    int64_t result;
    int16_t a_val = a;
    asm("cvt.s64.s8 %0, %1;" : "=l"(result) : "h"(a_val));
    return result;
}

__global__ void test_cvt_s64_s16_kernel(int16_t a, int64_t* result) { 
    int64_t temp = ptx_cvt_s64_s16(a);
    *result = temp;
}

void test_ptx_cvt_s64_s16(int16_t a, int64_t* result) {
    int64_t *d_result;
    cudaMalloc(&d_result, sizeof(int64_t));
    test_cvt_s64_s16_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int64_t ptx_cvt_s64_s16(int16_t a) {
    int64_t result;
    asm("cvt.s64.s16 %0, %1;" : "=l"(result) : "h"(a));
    return result;
}

__global__ void test_cvt_s64_s32_kernel(int32_t a, int64_t* result) { 
    int64_t temp = ptx_cvt_s64_s32(a);
    *result = temp;
}

void test_ptx_cvt_s64_s32(int32_t a, int64_t* result) {
    int64_t *d_result;
    cudaMalloc(&d_result, sizeof(int64_t));
    test_cvt_s64_s32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int64_t ptx_cvt_s64_s32(int32_t a) {
    int64_t result;
    asm("cvt.s64.s32 %0, %1;" : "=l"(result) : "r"(a));
    return result;
}

__global__ void test_cvt_u64_u8_kernel(uint8_t a, uint64_t* result) { 
    uint64_t temp = ptx_cvt_u64_u8(a);
    *result = temp;
}

void test_ptx_cvt_u64_u8(uint8_t a, uint64_t* result) {
    uint64_t *d_result;
    cudaMalloc(&d_result, sizeof(uint64_t));
    test_cvt_u64_u8_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint64_t ptx_cvt_u64_u8(uint8_t a) {
    uint64_t result;
    uint16_t a_val = a;
    asm("cvt.u64.u8 %0, %1;" : "=l"(result) : "h"(a_val));
    return result;
}

__global__ void test_cvt_u64_u16_kernel(uint16_t a, uint64_t* result) { 
    uint64_t temp = ptx_cvt_u64_u16(a);
    *result = temp;
}

void test_ptx_cvt_u64_u16(uint16_t a, uint64_t* result) {
    uint64_t *d_result;
    cudaMalloc(&d_result, sizeof(uint64_t));
    test_cvt_u64_u16_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint64_t ptx_cvt_u64_u16(uint16_t a) {
    uint64_t result;
    asm("cvt.u64.u16 %0, %1;" : "=l"(result) : "h"(a));
    return result;
}

__global__ void test_cvt_u64_u32_kernel(uint32_t a, uint64_t* result) { 
    uint64_t temp = ptx_cvt_u64_u32(a);
    *result = temp;
}

void test_ptx_cvt_u64_u32(uint32_t a, uint64_t* result) {
    uint64_t *d_result;
    cudaMalloc(&d_result, sizeof(uint64_t));
    test_cvt_u64_u32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint64_t ptx_cvt_u64_u32(uint32_t a) {
    uint64_t result;
    asm("cvt.u64.u32 %0, %1;" : "=l"(result) : "r"(a));
    return result;
}

// Float to integer conversions
__global__ void test_cvt_s32_f32_kernel(float a, int32_t* result) { 
    int32_t temp = ptx_cvt_s32_f32(a);
    *result = temp;
}

void test_ptx_cvt_s32_f32(float a, int32_t* result) {
    int32_t *d_result;
    cudaMalloc(&d_result, sizeof(int32_t));
    test_cvt_s32_f32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int32_t ptx_cvt_s32_f32(float a) {
    int32_t result;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(result) : "f"(a));
    return result;
}

__global__ void test_cvt_u32_f32_kernel(float a, uint32_t* result) { 
    uint32_t temp = ptx_cvt_u32_f32(a);
    *result = temp;
}

void test_ptx_cvt_u32_f32(float a, uint32_t* result) {
    uint32_t *d_result;
    cudaMalloc(&d_result, sizeof(uint32_t));
    test_cvt_u32_f32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint32_t ptx_cvt_u32_f32(float a) {
    uint32_t result;
    asm("cvt.rni.u32.f32 %0, %1;" : "=r"(result) : "f"(a));
    return result;
}

__global__ void test_cvt_s64_f64_kernel(double a, int64_t* result) { 
    int64_t temp = ptx_cvt_s64_f64(a);
    *result = temp;
}

void test_ptx_cvt_s64_f64(double a, int64_t* result) {
    int64_t *d_result;
    cudaMalloc(&d_result, sizeof(int64_t));
    test_cvt_s64_f64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int64_t ptx_cvt_s64_f64(double a) {
    int64_t result;
    asm("cvt.rni.s64.f64 %0, %1;" : "=l"(result) : "d"(a));
    return result;
}

__global__ void test_cvt_u64_f64_kernel(double a, uint64_t* result) { 
    uint64_t temp = ptx_cvt_u64_f64(a);
    *result = temp;
}

void test_ptx_cvt_u64_f64(double a, uint64_t* result) {
    uint64_t *d_result;
    cudaMalloc(&d_result, sizeof(uint64_t));
    test_cvt_u64_f64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint64_t ptx_cvt_u64_f64(double a) {
    uint64_t result;
    asm("cvt.rni.u64.f64 %0, %1;" : "=l"(result) : "d"(a));
    return result;
}

__global__ void test_cvt_s32_f64_kernel(double a, int32_t* result) { 
    int32_t temp = ptx_cvt_s32_f64(a);
    *result = temp;
}

void test_ptx_cvt_s32_f64(double a, int32_t* result) {
    int32_t *d_result;
    cudaMalloc(&d_result, sizeof(int32_t));
    test_cvt_s32_f64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ int32_t ptx_cvt_s32_f64(double a) {
    int32_t result;
    asm("cvt.rni.s32.f64 %0, %1;" : "=r"(result) : "d"(a));
    return result;
}

__global__ void test_cvt_u32_f64_kernel(double a, uint32_t* result) { 
    uint32_t temp = ptx_cvt_u32_f64(a);
    *result = temp;
}

void test_ptx_cvt_u32_f64(double a, uint32_t* result) {
    uint32_t *d_result;
    cudaMalloc(&d_result, sizeof(uint32_t));
    test_cvt_u32_f64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint32_t ptx_cvt_u32_f64(double a) {
    uint32_t result;
    asm("cvt.rni.u32.f64 %0, %1;" : "=r"(result) : "d"(a));
    return result;
}

// Integer to float conversions
__global__ void test_cvt_f32_s32_kernel(int32_t a, float* result) { 
    float temp = ptx_cvt_f32_s32(a);
    *result = temp;
}

void test_ptx_cvt_f32_s32(int32_t a, float* result) {
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));
    test_cvt_f32_s32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ float ptx_cvt_f32_s32(int32_t a) {
    float result;
    asm("cvt.f32.s32.rn %0, %1;" : "=f"(result) : "r"(a));
    return result;
}

__global__ void test_cvt_f32_u32_kernel(uint32_t a, float* result) { 
    float temp = ptx_cvt_f32_u32(a);
    *result = temp;
}

void test_ptx_cvt_f32_u32(uint32_t a, float* result) {
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));
    test_cvt_f32_u32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ float ptx_cvt_f32_u32(uint32_t a) {
    float result;
    asm("cvt.f32.u32.rn %0, %1;" : "=f"(result) : "r"(a));
    return result;
}

__global__ void test_cvt_f64_s64_kernel(int64_t a, double* result) { 
    double temp = ptx_cvt_f64_s64(a);
    *result = temp;
}

void test_ptx_cvt_f64_s64(int64_t a, double* result) {
    double *d_result;
    cudaMalloc(&d_result, sizeof(double));
    test_cvt_f64_s64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ double ptx_cvt_f64_s64(int64_t a) {
    double result;
    asm("cvt.f64.s64.rn %0, %1;" : "=d"(result) : "l"(a));
    return result;
}

__global__ void test_cvt_f64_u64_kernel(uint64_t a, double* result) { 
    double temp = ptx_cvt_f64_u64(a);
    *result = temp;
}

void test_ptx_cvt_f64_u64(uint64_t a, double* result) {
    double *d_result;
    cudaMalloc(&d_result, sizeof(double));
    test_cvt_f64_u64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ double ptx_cvt_f64_u64(uint64_t a) {
    double result;
    asm("cvt.f64.u64.rn %0, %1;" : "=d"(result) : "l"(a));
    return result;
}

__global__ void test_cvt_f32_s64_kernel(int64_t a, float* result) { 
    float temp = ptx_cvt_f32_s64(a);
    *result = temp;
}

void test_ptx_cvt_f32_s64(int64_t a, float* result) {
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));
    test_cvt_f32_s64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ float ptx_cvt_f32_s64(int64_t a) {
    float result;
    asm("cvt.f32.s64.rn %0, %1;" : "=f"(result) : "l"(a));
    return result;
}

__global__ void test_cvt_f32_u64_kernel(uint64_t a, float* result) { 
    float temp = ptx_cvt_f32_u64(a);
    *result = temp;
}

void test_ptx_cvt_f32_u64(uint64_t a, float* result) {
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));
    test_cvt_f32_u64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ float ptx_cvt_f32_u64(uint64_t a) {
    float result;
    asm("cvt.f32.u64.rn %0, %1;" : "=f"(result) : "l"(a));
    return result;
}

// Float to float conversions
__global__ void test_cvt_f32_f64_kernel(double a, float* result) { 
    float temp = ptx_cvt_f32_f64(a);
    *result = temp;
}

void test_ptx_cvt_f32_f64(double a, float* result) {
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));
    test_cvt_f32_f64_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ float ptx_cvt_f32_f64(double a) {
    float result;
    asm("cvt.f32.f64.rn %0, %1;" : "=f"(result) : "d"(a));
    return result;
}

__global__ void test_cvt_f64_f32_kernel(float a, double* result) { 
    double temp = ptx_cvt_f64_f32(a);
    *result = temp;
}

void test_ptx_cvt_f64_f32(float a, double* result) {
    double *d_result;
    cudaMalloc(&d_result, sizeof(double));
    test_cvt_f64_f32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ double ptx_cvt_f64_f32(float a) {
    double result;
    asm("cvt.f64.f32 %0, %1;" : "=d"(result) : "f"(a));
    return result;
}

// Half precision (f16) to float (f32) conversions
__global__ void test_cvt_f32_f16_kernel(__nv_half a, float* result) { 
    float temp = ptx_cvt_f32_f16(a);
    *result = temp;
}

void test_ptx_cvt_f32_f16(__nv_half a, float* result) {
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));
    test_cvt_f32_f16_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ float ptx_cvt_f32_f16(__nv_half a) {
    float result;
    __half half_val = (__half)a;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"((short&)half_val));
    return result;
}

__global__ void test_cvt_f16_f32_kernel(float a, __nv_half* result) { 
    __nv_half temp = ptx_cvt_f16_f32(a);
    *result = temp;
}

void test_ptx_cvt_f16_f32(float a, __nv_half* result) {
    __nv_half *d_result;
    cudaMalloc(&d_result, sizeof(__nv_half));
    test_cvt_f16_f32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(__nv_half), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ __nv_half ptx_cvt_f16_f32(float a) {
    __half result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"((short&)result) : "f"(a));
    return (__nv_half)result;
}

// Saturation conversions
__global__ void test_cvt_satu8_f32_kernel(float a, uint8_t* result) { 
    uint8_t temp = ptx_cvt_satu8_f32(a);
    *result = temp;
}

void test_ptx_cvt_satu8_f32(float a, uint8_t* result) {
    uint8_t *d_result;
    cudaMalloc(&d_result, sizeof(uint8_t));
    test_cvt_satu8_f32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint8_t ptx_cvt_satu8_f32(float a) {
    uint16_t result;
    asm("cvt.u8.f32.sat.rni %0, %1;" : "=h"(result) : "f"(a));
    return (uint8_t)result;
}

__global__ void test_cvt_satu16_f32_kernel(float a, uint16_t* result) { 
    uint16_t temp = ptx_cvt_satu16_f32(a);
    *result = temp;
}

void test_ptx_cvt_satu16_f32(float a, uint16_t* result) {
    uint16_t *d_result;
    cudaMalloc(&d_result, sizeof(uint16_t));
    test_cvt_satu16_f32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint16_t ptx_cvt_satu16_f32(float a) {
    uint16_t result;
    asm("cvt.sat.u16.f32.rni %0, %1;" : "=h"(result) : "f"(a));
    return result;
}

__global__ void test_cvt_satu32_f32_kernel(float a, uint32_t* result) { 
    uint32_t temp = ptx_cvt_satu32_f32(a);
    *result = temp;
}

void test_ptx_cvt_satu32_f32(float a, uint32_t* result) {
    uint32_t *d_result;
    cudaMalloc(&d_result, sizeof(uint32_t));
    test_cvt_satu32_f32_kernel<<<1, 1>>>(a, d_result);
    cudaMemcpy(result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__device__ uint32_t ptx_cvt_satu32_f32(float a) {
    uint32_t result;
    asm("cvt.sat.u32.f32.rni %0, %1;" : "=r"(result) : "f"(a));
    return result;
}
