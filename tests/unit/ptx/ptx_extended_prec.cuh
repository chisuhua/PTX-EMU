#ifndef PTX_EXTENDED_PREC_CUH
#define PTX_EXTENDED_PREC_CUH

#include <cuda_runtime.h>
#include <cstdint>

// --- Device-side PTX wrappers for extended precision integer operations ---
// 这些函数用于测试 PTX-EMU 的 addc/subc/mul24/mul 指令模拟
#ifdef __CUDA_ARCH__

// 使用显式 uint32_t 中间变量和 volatile 确保正确写入
__device__ __forceinline__ uint32_t ptx_addc_u32(uint32_t a, uint32_t b, bool carry_in, bool* carry_out) {
    uint64_t sum = static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
    if (carry_in) {
        sum += 1ULL;
    }
    uint32_t result = static_cast<uint32_t>(sum);
    // 使用 volatile 确保值被正确写入
    volatile uint32_t carry_val = (sum > 0xFFFFFFFFULL) ? 1U : 0U;
    *reinterpret_cast<volatile uint32_t*>(carry_out) = carry_val;
    return result;
}

__device__ __forceinline__ uint32_t ptx_subc_u32(uint32_t a, uint32_t b, bool borrow_in, bool* borrow_out) {
    uint64_t minuend = static_cast<uint64_t>(a);
    uint64_t subtrahend = static_cast<uint64_t>(b);
    if (borrow_in) {
        subtrahend += 1ULL;
    }
    uint64_t diff = minuend - subtrahend;
    uint32_t result = static_cast<uint32_t>(diff);
    // 借位条件：minuend < subtrahend
    volatile uint32_t borrow_val = (minuend < subtrahend) ? 1U : 0U;
    *reinterpret_cast<volatile uint32_t*>(borrow_out) = borrow_val;
    return result;
}

#else
// Host-side stub
__device__ __forceinline__ uint32_t ptx_addc_u32(uint32_t a, uint32_t b, bool carry_in, bool* carry_out) {
    uint64_t sum = static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
    if (carry_in) {
        sum += 1ULL;
    }
    uint32_t result = static_cast<uint32_t>(sum);
    volatile uint32_t carry_val = (sum > 0xFFFFFFFFULL) ? 1U : 0U;
    *reinterpret_cast<volatile uint32_t*>(carry_out) = carry_val;
    return result;
}
__device__ __forceinline__ uint32_t ptx_subc_u32(uint32_t a, uint32_t b, bool borrow_in, bool* borrow_out) {
    uint64_t minuend = static_cast<uint64_t>(a);
    uint64_t subtrahend = static_cast<uint64_t>(b);
    if (borrow_in) {
        subtrahend += 1ULL;
    }
    uint64_t diff = minuend - subtrahend;
    uint32_t result = static_cast<uint32_t>(diff);
    volatile uint32_t borrow_val = (minuend < subtrahend) ? 1U : 0U;
    *reinterpret_cast<volatile uint32_t*>(borrow_out) = borrow_val;
    return result;
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