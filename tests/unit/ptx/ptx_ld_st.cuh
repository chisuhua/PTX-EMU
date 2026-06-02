#ifndef PTX_LD_ST_CUH
#define PTX_LD_ST_CUH

#include <cuda_runtime.h>
#include <cstdint>

// --- Device-side PTX wrappers (inline assembly) for LD/ST operations ---

// Global memory load operations
template<typename T>
__device__ __forceinline__ void ptx_ld_global(T* result, T* addr) {
    *result = *addr;
}

// Global memory store operations
template<typename T>
__device__ __forceinline__ void ptx_st_global(T* addr, T value) {
    *addr = value;
}

// Load operations for various types
__device__ __forceinline__ void ptx_ld_u8(uint8_t* result, uint8_t* addr) {
    unsigned int temp;
    asm volatile("ld.global.u8 %0, [%1];" : "=r"(temp) : "l"(addr));
    *result = static_cast<uint8_t>(temp);
}

__device__ __forceinline__ void ptx_ld_u16(uint16_t* result, uint16_t* addr) {
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(*result) : "l"(addr));
}

__device__ __forceinline__ void ptx_ld_u32(uint32_t* result, uint32_t* addr) {
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(*result) : "l"(addr));
}

__device__ __forceinline__ void ptx_ld_u64(uint64_t* result, uint64_t* addr) {
    asm volatile("ld.global.u64 %0, [%1];" : "=l"(*result) : "l"(addr));
}

__device__ __forceinline__ void ptx_ld_f32(float* result, float* addr) {
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(*result) : "l"(addr));
}

__device__ __forceinline__ void ptx_ld_f64(double* result, double* addr) {
    asm volatile("ld.global.f64 %0, [%1];" : "=d"(*result) : "l"(addr));
}

// Store operations for various types
__device__ __forceinline__ void ptx_st_u8(uint8_t* addr, uint8_t value) {
    asm volatile("st.global.u8 [%0], %1;" :: "l"(addr), "r"((unsigned int)(value)));
}

__device__ __forceinline__ void ptx_st_u16(uint16_t* addr, uint16_t value) {
    asm volatile("st.global.u16 [%0], %1;" :: "l"(addr), "h"(value));
}

__device__ __forceinline__ void ptx_st_u32(uint32_t* addr, uint32_t value) {
    asm volatile("st.global.u32 [%0], %1;" :: "l"(addr), "r"(value));
}

__device__ __forceinline__ void ptx_st_u64(uint64_t* addr, uint64_t value) {
    asm volatile("st.global.u64 [%0], %1;" :: "l"(addr), "l"(value));
}

__device__ __forceinline__ void ptx_st_f32(float* addr, float value) {
    asm volatile("st.global.f32 [%0], %1;" :: "l"(addr), "f"(value));
}

__device__ __forceinline__ void ptx_st_f64(double* addr, double value) {
    asm volatile("st.global.f64 [%0], %1;" :: "l"(addr), "d"(value));
}

// Shared memory operations (using local arrays to simulate shared memory)
__device__ __forceinline__ void ptx_ld_shared_u32(uint32_t* result, uint32_t* addr) {
    *result = *addr;
}

__device__ __forceinline__ void ptx_st_shared_u32(uint32_t* addr, uint32_t value) {
    *addr = value;
}

// Kernel declarations
__global__ void ld_u8_kernel(uint8_t* input, uint8_t* output);
__global__ void ld_u16_kernel(uint16_t* input, uint16_t* output);
__global__ void ld_u32_kernel(uint32_t* input, uint32_t* output);
__global__ void ld_u64_kernel(uint64_t* input, uint64_t* output);
__global__ void ld_f32_kernel(float* input, float* output);
__global__ void ld_f64_kernel(double* input, double* output);
__global__ void st_u8_kernel(uint8_t* output, uint8_t value);
__global__ void st_u16_kernel(uint16_t* output, uint16_t value);
__global__ void st_u32_kernel(uint32_t* output, uint32_t value);
__global__ void st_u64_kernel(uint64_t* output, uint64_t value);
__global__ void st_f32_kernel(float* output, float value);
__global__ void st_f64_kernel(double* output, double value);

// 函数声明
void test_ptx_ld_u8(uint8_t value, uint8_t* result);
void test_ptx_ld_u16(uint16_t value, uint16_t* result);
void test_ptx_ld_u32(uint32_t value, uint32_t* result);
void test_ptx_ld_u64(uint64_t value, uint64_t* result);
void test_ptx_ld_f32(float value, float* result);
void test_ptx_ld_f64(double value, double* result);
void test_ptx_st_u8(uint8_t* addr, uint8_t value, uint8_t* result);
void test_ptx_st_u16(uint16_t* addr, uint16_t value, uint16_t* result);
void test_ptx_st_u32(uint32_t* addr, uint32_t value, uint32_t* result);
void test_ptx_st_u64(uint64_t* addr, uint64_t value, uint64_t* result);
void test_ptx_st_f32(float* addr, float value, float* result);
void test_ptx_st_f64(double* addr, double value, double* result);
void test_ptx_shared_load_store(uint32_t* result);
void test_ptx_shared_store_only(uint32_t* result);

#endif // PTX_LD_ST_CUH