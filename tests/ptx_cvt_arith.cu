#include "ptx_cvt_arith.cuh"

// Integer to integer conversions

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

// 删除重复定义的kernel和函数
// __global__ void test_cvt_s8_s32_kernel(int32_t a, int8_t* result) {
//     int8_t temp = ptx_cvt_s8_s32(a);
//     *result = temp;
// }
// 
// void test_ptx_cvt_s8_s32(int32_t a, int8_t* result) {
//     int8_t *d_result;
//     cudaMalloc(&d_result, sizeof(int8_t));
//     test_cvt_s8_s32_kernel<<<1, 1>>>(a, d_result);
//     cudaMemcpy(result, d_result, sizeof(int8_t), cudaMemcpyDeviceToHost);
//     cudaFree(d_result);
// }

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

// 删除重复定义的kernel和函数
// __global__ void test_cvt_u8_u16_kernel(uint16_t a, uint8_t* result) {
//     uint8_t temp = ptx_cvt_u8_u16(a);
//     *result = temp;
// }
// 
// void test_ptx_cvt_u8_u16(uint16_t a, uint8_t* result) {
//     uint8_t *d_result;
//     cudaMalloc(&d_result, sizeof(uint8_t));
//     test_cvt_u8_u16_kernel<<<1, 1>>>(a, d_result);
//     cudaMemcpy(result, d_result, sizeof(uint8_t), cudaMemcpyDeviceToHost);
//     cudaFree(d_result);
// }

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint8_t ptx_cvt_u8_u32(uint32_t a) {
//     return ptx_cvt_u8_u32(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint8_t ptx_cvt_u8_u64(uint64_t a) {
//     return ptx_cvt_u8_u64(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ int16_t ptx_cvt_s16_s8(int8_t a) {
//     return ptx_cvt_s16_s8(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ int16_t ptx_cvt_s16_s32(int32_t a) {
//     return ptx_cvt_s16_s32(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ int16_t ptx_cvt_s16_s64(int64_t a) {
//     return ptx_cvt_s16_s64(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint16_t ptx_cvt_u16_u8(uint8_t a) {
//     return ptx_cvt_u16_u8(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint16_t ptx_cvt_u16_u32(uint32_t a) {
//     return ptx_cvt_u16_u32(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint16_t ptx_cvt_u16_u64(uint64_t a) {
//     return ptx_cvt_u16_u64(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ int32_t ptx_cvt_s32_s8(int8_t a) {
//     return ptx_cvt_s32_s8(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ int32_t ptx_cvt_s32_s16(int16_t a) {
//     return ptx_cvt_s32_s16(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ int32_t ptx_cvt_s32_s64(int64_t a) {
//     return ptx_cvt_s32_s64(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint32_t ptx_cvt_u32_u8(uint8_t a) {
//     return ptx_cvt_u32_u8(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint32_t ptx_cvt_u32_u16(uint16_t a) {
//     return ptx_cvt_u32_u16(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint32_t ptx_cvt_u32_u64(uint64_t a) {
//     return ptx_cvt_u32_u64(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ int64_t ptx_cvt_s64_s8(int8_t a) {
//     return ptx_cvt_s64_s8(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ int64_t ptx_cvt_s64_s16(int16_t a) {
//     return ptx_cvt_s64_s16(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ int64_t ptx_cvt_s64_s32(int32_t a) {
//     return ptx_cvt_s64_s32(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint64_t ptx_cvt_u64_u8(uint8_t a) {
//     return ptx_cvt_u64_u8(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint64_t ptx_cvt_u64_u16(uint16_t a) {
//     return ptx_cvt_u64_u16(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint64_t ptx_cvt_u64_u32(uint32_t a) {
//     return ptx_cvt_u64_u32(a);
// }

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

// Float to integer conversions
// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ int32_t ptx_cvt_s32_f32(float a) {
//     return ptx_cvt_s32_f32(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint32_t ptx_cvt_u32_f32(float a) {
//     return ptx_cvt_u32_f32(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ int64_t ptx_cvt_s64_f64(double a) {
//     return ptx_cvt_s64_f64(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint64_t ptx_cvt_u64_f64(double a) {
//     return ptx_cvt_u64_f64(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ int32_t ptx_cvt_s32_f64(double a) {
//     return ptx_cvt_s32_f64(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint32_t ptx_cvt_u32_f64(double a) {
//     return ptx_cvt_u32_f64(a);
// }

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

// Integer to float conversions
// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ float ptx_cvt_f32_s32(int32_t a) {
//     return ptx_cvt_f32_s32(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ float ptx_cvt_f32_u32(uint32_t a) {
//     return ptx_cvt_f32_u32(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ double ptx_cvt_f64_s64(int64_t a) {
//     return ptx_cvt_f64_s64(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ double ptx_cvt_f64_u64(uint64_t a) {
//     return ptx_cvt_f64_u64(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ float ptx_cvt_f32_s64(int64_t a) {
//     return ptx_cvt_f32_s64(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ float ptx_cvt_f32_u64(uint64_t a) {
//     return ptx_cvt_f32_u64(a);
// }

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

// Float to float conversions
// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ float ptx_cvt_f32_f64(double a) {
//     return ptx_cvt_f32_f64(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ double ptx_cvt_f64_f32(float a) {
//     return ptx_cvt_f64_f32(a);
// }

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

// Saturation conversions
// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint8_t ptx_cvt_satu8_f32(float a) {
//     return ptx_cvt_satu8_f32(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint16_t ptx_cvt_satu16_f32(float a) {
//     return ptx_cvt_satu16_f32(a);
// }

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

// 删除重复定义的device函数，因为它们已经在头文件中定义了
// __device__ uint32_t ptx_cvt_satu32_f32(float a) {
//     return ptx_cvt_satu32_f32(a);
// }

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