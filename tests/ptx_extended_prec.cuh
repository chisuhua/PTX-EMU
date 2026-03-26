#ifndef PTX_EXTENDED_PREC_CUH
#define PTX_EXTENDED_PREC_CUH

#include <cuda_runtime.h>
#include <cstdint>

// --- Device-side PTX wrappers (inline assembly) for extended precision integer operations ---
// 使用纯 C++ 实现来避免 NVCC 内联汇编的寄存器分配问题
// 注意：NVCC 会优化掉这些函数中的 addc/subc 指令，
// 因此这个实现不会真正测试模拟器的 addc/subc 指令处理器
#ifdef __CUDA_ARCH__
__device__ __forceinline__ uint32_t ptx_addc_u32(uint32_t a, uint32_t b, bool carry_in, bool* carry_out) {
    uint64_t result64 = static_cast<uint64_t>(a) + static_cast<uint64_t>(b) + (carry_in ? 1ULL : 0ULL);
    uint32_t result = static_cast<uint32_t>(result64);
    *carry_out = (result64 > 0xFFFFFFFFULL);
    return result;
}

__device__ __forceinline__ uint32_t ptx_subc_u32(uint32_t a, uint32_t b, bool borrow_in, bool* borrow_out) {
    uint64_t a64 = a;
    uint64_t b64 = b;
    uint64_t result64 = a64 - b64 - (borrow_in ? 1ULL : 0ULL);
    uint32_t result = static_cast<uint32_t>(result64);
    *borrow_out = (a64 < b64 + (borrow_in ? 1ULL : 0ULL));
    return result;
}
#else
// Host-side stub
__device__ __forceinline__ uint32_t ptx_addc_u32(uint32_t a, uint32_t b, bool carry_in, bool* carry_out) {
    *carry_out = false;
    return a + b + (carry_in ? 1 : 0);
}
__device__ __forceinline__ uint32_t ptx_subc_u32(uint32_t a, uint32_t b, bool borrow_in, bool* borrow_out) {
    *borrow_out = false;
    return a - b - (borrow_in ? 1 : 0);
}
#endif

__device__ __forceinline__ uint32_t ptx_mul24_lo_u32(uint32_t a, uint32_t b) {
    uint32_t res;
    asm("mul24.lo.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ uint32_t ptx_mul24_hi_u32(uint32_t a, uint32_t b) {
    uint32_t res;
    asm("mul24.hi.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ uint32_t ptx_mul_lo_u32(uint32_t a, uint32_t b) {
    uint32_t res;
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ uint32_t ptx_mul_hi_u32(uint32_t a, uint32_t b) {
    uint32_t res;
    asm("mul.hi.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ uint32_t ptx_mul_wide_u32(uint32_t a, uint32_t b) {
    uint64_t res;
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(res) : "r"(a), "r"(b));
    return (uint32_t)res;  // 返回低32位
}

__device__ __forceinline__ int32_t ptx_mul24_lo_s32(int32_t a, int32_t b) {
    int32_t res;
    asm("mul24.lo.s32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ int32_t ptx_mul24_hi_s32(int32_t a, int32_t b) {
    int32_t res;
    asm("mul24.hi.s32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ int32_t ptx_mul_lo_s32(int32_t a, int32_t b) {
    int32_t res;
    asm("mul.lo.s32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ int32_t ptx_mul_hi_s32(int32_t a, int32_t b) {
    int32_t res;
    asm("mul.hi.s32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    return res;
}

__device__ __forceinline__ uint64_t ptx_mul_wide_u32_to_u64(uint32_t a, uint32_t b) {
    uint64_t res;
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(res) : "r"(a), "r"(b));
    return res;
}

// --- Host-side wrapper functions ---
void test_ptx_addc_u32(uint32_t a, uint32_t b, bool carry_in, uint32_t* result, bool* carry_out);
void test_ptx_subc_u32(uint32_t a, uint32_t b, bool borrow_in, uint32_t* result, bool* borrow_out);
void test_ptx_mul24_lo_u32(uint32_t a, uint32_t b, uint32_t* result);
void test_ptx_mul24_hi_u32(uint32_t a, uint32_t b, uint32_t* result);
void test_ptx_mul_lo_u32(uint32_t a, uint32_t b, uint32_t* result);
void test_ptx_mul_hi_u32(uint32_t a, uint32_t b, uint32_t* result);
void test_ptx_mul_wide_u32(uint32_t a, uint32_t b, uint32_t* result);
void test_ptx_mul24_lo_s32(int32_t a, int32_t b, int32_t* result);
void test_ptx_mul24_hi_s32(int32_t a, int32_t b, int32_t* result);
void test_ptx_mul_lo_s32(int32_t a, int32_t b, int32_t* result);
void test_ptx_mul_hi_s32(int32_t a, int32_t b, int32_t* result);
void test_ptx_mul_wide_u32_to_u64(uint32_t a, uint32_t b, uint64_t* result);

#endif // PTX_EXTENDED_PREC_CUH