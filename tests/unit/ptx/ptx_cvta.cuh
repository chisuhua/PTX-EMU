#ifndef PTX_CVTA_CUH
#define PTX_CVTA_CUH

#include <cuda_runtime.h>
#include <cstdint>

// 内核函数声明和定义
inline __global__ void kernel_test_ptx_cvta_to_global_u64(uint64_t input_addr, uint64_t *result) {
    // CVTA to global address space conversion
    uint64_t converted_addr;
    asm("cvta.to.global.u64 %0, %1;" : "=l"(converted_addr) : "l"(input_addr));
    *result = converted_addr;
}

inline __global__ void kernel_test_ptx_cvta_to_local_u64(uint64_t input_addr, uint64_t *result) {
    // CVTA to local address space conversion
    uint64_t converted_addr;
    asm("cvta.to.local.u64 %0, %1;" : "=l"(converted_addr) : "l"(input_addr));
    *result = converted_addr;
}

inline __global__ void kernel_test_ptx_cvta_to_param_u64(uint64_t input_addr, uint64_t *result) {
    // CVTA to param address space conversion
    uint64_t converted_addr;
    asm("cvta.to.param.u64 %0, %1;" : "=l"(converted_addr) : "l"(input_addr));
    *result = converted_addr;
}

// 为32位地址转换使用简单的转换方式
// __global__ void kernel_test_ptx_cvta_to_shared_u32(uint32_t input_addr, uint32_t *result) {
//     // CVTA to shared address space conversion for u32
//     uint32_t converted_addr;
//     asm("cvta.to.shared.u32 %0, %1;" : "=r"(converted_addr) : "r"(input_addr));
//     *result = converted_addr;
// }

// __global__ void kernel_test_ptx_cvta_to_global_u32(uint32_t input_addr, uint32_t *result) {
//     // CVTA to global address space conversion for u32
//     uint32_t converted_addr;
//     asm("cvta.to.global.u32 %0, %1;" : "=r"(converted_addr) : "r"(input_addr));
//     *result = converted_addr;
// }

inline __global__ void kernel_test_ptx_cvta_const_u64(uint64_t input_addr, uint64_t *result) {
    // CVTA const address space conversion
    uint64_t converted_addr;
    asm("cvta.const.u64 %0, %1;" : "=l"(converted_addr) : "l"(input_addr));
    *result = converted_addr;
}

#endif // PTX_CVTA_CUH