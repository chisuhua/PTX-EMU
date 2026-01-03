#ifndef PTX_EXTENDED_PREC_CUH
#define PTX_EXTENDED_PREC_CUH

#include <cuda_runtime.h>
#include <cstdint>

// --- Device-side PTX wrappers (inline assembly) for extended precision integer operations ---
__device__ __forceinline__ uint32_t ptx_addc_u32(uint32_t a, uint32_t b, bool carry_in, bool* carry_out) {
    uint32_t res;
    if (carry_in) {
        asm("add.cc.u32 %%r0, 0xFFFFFFFF, 1;");  // 设置进位为1: 0xFFFFFFFF + 1 = 0, carry=1
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));  // 执行 a + b + 1
    } else {
        asm("add.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));   // 执行 a + b + 0
    }
    uint32_t temp_carry;
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(temp_carry) : "r"(0), "r"(0));  // 获取当前进位状态
    *carry_out = (bool)temp_carry;
    return res;
}

__device__ __forceinline__ uint32_t ptx_subc_u32(uint32_t a, uint32_t b, bool borrow_in, bool* borrow_out) {
    uint32_t res;
    if (borrow_in) {
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    } else {
        asm("sub.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
    }
    uint32_t temp_borrow;
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(temp_borrow) : "r"(0), "r"(0));
    *borrow_out = (bool)temp_borrow;
    return res;
}

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