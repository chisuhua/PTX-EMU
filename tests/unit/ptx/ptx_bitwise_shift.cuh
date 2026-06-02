#ifndef PTX_BITWISE_SHIFT_CUH
#define PTX_BITWISE_SHIFT_CUH

#include <cuda_runtime.h>
#include <cstdint>

// --- Device-side PTX wrappers (inline assembly) for bitwise operations ---
__device__ __forceinline__ uint32_t ptx_and_b32(uint32_t a, uint32_t b) {
    uint32_t res;
    asm("and.b32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ uint32_t ptx_or_b32(uint32_t a, uint32_t b) {
    uint32_t res;
    asm("or.b32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ uint32_t ptx_xor_b32(uint32_t a, uint32_t b) {
    uint32_t res;
    asm("xor.b32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ uint32_t ptx_not_b32(uint32_t a) {
    uint32_t res;
    asm("not.b32 %0, %1;" : "=r"(res) : "r"(a));
    return res;
}

__device__ __forceinline__ uint64_t ptx_and_b64(uint64_t a, uint64_t b) {
    uint64_t res;
    asm("and.b64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));
    return res;
}

__device__ __forceinline__ uint64_t ptx_or_b64(uint64_t a, uint64_t b) {
    uint64_t res;
    asm("or.b64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));
    return res;
}

__device__ __forceinline__ uint64_t ptx_xor_b64(uint64_t a, uint64_t b) {
    uint64_t res;
    asm("xor.b64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));
    return res;
}

__device__ __forceinline__ uint64_t ptx_not_b64(uint64_t a) {
    uint64_t res;
    asm("not.b64 %0, %1;" : "=l"(res) : "l"(a));
    return res;
}

// --- Device-side PTX wrappers for shift operations ---
__device__ __forceinline__ uint32_t ptx_shl_b32(uint32_t a, uint32_t b) {
    return a << (b & 0x1f);  // mask to 5 bits for 32-bit shifts
}

__device__ __forceinline__ uint32_t ptx_shr_b32(uint32_t a, uint32_t b) {
    return a >> (b & 0x1f);  // mask to 5 bits for 32-bit shifts
}

__device__ __forceinline__ uint64_t ptx_shl_b64(uint64_t a, uint64_t b) {
    return a << (b & 0x3f);  // mask to 6 bits for 64-bit shifts
}

__device__ __forceinline__ uint64_t ptx_shr_b64(uint64_t a, uint64_t b) {
    return a >> (b & 0x3f);  // mask to 6 bits for 64-bit shifts
}

// --- Host-side wrapper functions for uint32 ---
void test_ptx_and_b32(uint32_t a, uint32_t b, uint32_t* result);
void test_ptx_or_b32(uint32_t a, uint32_t b, uint32_t* result);
void test_ptx_xor_b32(uint32_t a, uint32_t b, uint32_t* result);
void test_ptx_not_b32(uint32_t a, uint32_t* result);
void test_ptx_shl_b32(uint32_t a, uint32_t b, uint32_t* result);
void test_ptx_shr_b32(uint32_t a, uint32_t b, uint32_t* result);

// --- Host-side wrapper functions for uint64 ---
void test_ptx_and_b64(uint64_t a, uint64_t b, uint64_t* result);
void test_ptx_or_b64(uint64_t a, uint64_t b, uint64_t* result);
void test_ptx_xor_b64(uint64_t a, uint64_t b, uint64_t* result);
void test_ptx_not_b64(uint64_t a, uint64_t* result);
void test_ptx_shl_b64(uint64_t a, uint64_t b, uint64_t* result);
void test_ptx_shr_b64(uint64_t a, uint64_t b, uint64_t* result);

#endif // PTX_BITWISE_SHIFT_CUH