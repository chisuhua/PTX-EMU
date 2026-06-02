#ifndef PTX_OPS_CUH
#define PTX_OPS_CUH

#include <cuda_runtime.h>
#include <cstdint>

// --- Device-side PTX wrappers (inline assembly) ---
__device__ __forceinline__ int ptx_add_s32(int a, int b) {
    int res;
    asm("add.s32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ int ptx_sub_s32(int a, int b) {
    int res;
    asm("sub.s32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ int ptx_mul_s32(int a, int b) {
    int res;
    asm("mul.lo.s32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ int ptx_mul24_s32(int a, int b) {
    int res;
    asm("mul24.lo.s32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ int ptx_mad_s32(int a, int b, int c) {
    int res;
    asm("mad.lo.s32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}

__device__ __forceinline__ int ptx_mad24_s32(int a, int b, int c) {
    int res;
    asm("mad24.lo.s32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}

__device__ __forceinline__ int ptx_div_s32(int a, int b) {
    int res;
    asm("div.s32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ int ptx_rem_s32(int a, int b) {
    int res;
    asm("rem.s32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ int ptx_abs_s32(int a) {
    int res;
    asm("abs.s32 %0, %1;" : "=r"(res) : "r"(a));
    return res;
}

__device__ __forceinline__ int ptx_neg_s32(int a) {
    int res;
    asm("neg.s32 %0, %1;" : "=r"(res) : "r"(a));
    return res;
}

__device__ __forceinline__ int ptx_min_s32(int a, int b) {
    int res;
    asm("min.s32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ int ptx_max_s32(int a, int b) {
    int res;
    asm("max.s32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ int ptx_popc_u32(unsigned int a) {
    int res;
    asm("popc.b32 %0, %1;" : "=r"(res) : "r"(a));
    return res;
}

__device__ __forceinline__ int ptx_clz_u32(unsigned int a) {
    int res;
    asm("clz.b32 %0, %1;" : "=r"(res) : "r"(a));
    return res;
}

// __device__ __forceinline__ int ptx_bfind_s32(int a) {
//     int res;
//     asm("bfind.reverse.s32 %0, %1;" : "=r"(res) : "r"(a));
//     return res;
// }

// Note: fns is stateful and warp-wide; omitted for simplicity in basic test

// --- Host-side wrapper functions ---
void test_ptx_add_s32(int a, int b, int* result);
void test_ptx_sub_s32(int a, int b, int* result);
void test_ptx_mul_s32(int a, int b, int* result);
void test_ptx_mul24_s32(int a, int b, int* result);
void test_ptx_mad_s32(int a, int b, int c, int* result);
void test_ptx_mad24_s32(int a, int b, int c, int* result);
void test_ptx_div_s32(int a, int b, int* result);
void test_ptx_rem_s32(int a, int b, int* result);
void test_ptx_abs_s32(int a, int* result);
void test_ptx_neg_s32(int a, int* result);
void test_ptx_min_s32(int a, int b, int* result);
void test_ptx_max_s32(int a, int b, int* result);
void test_ptx_popc_u32(unsigned int a, int* result);
void test_ptx_clz_u32(unsigned int a, int* result);
void test_ptx_bfind_s32(int a, int* result);

#endif // PTX_OPS_CUH