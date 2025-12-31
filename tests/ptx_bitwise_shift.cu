// ptx_bitwise_shift.cu
#include "ptx_bitwise_shift.cuh"
#include <stdexcept>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error("CUDA error: "/* + std::string(cudaGetErrorString(err))*/); \
    } \
} while(0)

template<typename Kernel, typename... Args>
void launch_and_copy_back_uint32(uint32_t* d_result, uint32_t* h_result, Kernel kernel, Args... args) {
    kernel<<<1, 1>>>(args..., d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

template<typename Kernel, typename... Args>
void launch_and_copy_back_uint64(uint64_t* d_result, uint64_t* h_result, Kernel kernel, Args... args) {
    kernel<<<1, 1>>>(args..., d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
}

// Kernels for bitwise operations
__global__ void and_kernel_u32(uint32_t a, uint32_t b, uint32_t* res) { *res = ptx_and_b32(a, b); }
__global__ void or_kernel_u32(uint32_t a, uint32_t b, uint32_t* res) { *res = ptx_or_b32(a, b); }
__global__ void xor_kernel_u32(uint32_t a, uint32_t b, uint32_t* res) { *res = ptx_xor_b32(a, b); }
__global__ void not_kernel_u32(uint32_t a, uint32_t* res) { *res = ptx_not_b32(a); }

__global__ void and_kernel_u64(uint64_t a, uint64_t b, uint64_t* res) { *res = ptx_and_b64(a, b); }
__global__ void or_kernel_u64(uint64_t a, uint64_t b, uint64_t* res) { *res = ptx_or_b64(a, b); }
__global__ void xor_kernel_u64(uint64_t a, uint64_t b, uint64_t* res) { *res = ptx_xor_b64(a, b); }
__global__ void not_kernel_u64(uint64_t a, uint64_t* res) { *res = ptx_not_b64(a); }

// Kernels for shift operations
__global__ void shl_kernel_u32(uint32_t a, uint32_t b, uint32_t* res) { *res = ptx_shl_b32(a, b); }
__global__ void shr_kernel_u32(uint32_t a, uint32_t b, uint32_t* res) { *res = ptx_shr_b32(a, b); }

__global__ void shl_kernel_u64(uint64_t a, uint64_t b, uint64_t* res) { *res = ptx_shl_b64(a, b); }
__global__ void shr_kernel_u64(uint64_t a, uint64_t b, uint64_t* res) { *res = ptx_shr_b64(a, b); }

// Host wrappers for uint32 operations
void test_ptx_and_b32(uint32_t a, uint32_t b, uint32_t* result) {
    uint32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint32_t)));
    launch_and_copy_back_uint32(d_res, result, and_kernel_u32, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_or_b32(uint32_t a, uint32_t b, uint32_t* result) {
    uint32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint32_t)));
    launch_and_copy_back_uint32(d_res, result, or_kernel_u32, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_xor_b32(uint32_t a, uint32_t b, uint32_t* result) {
    uint32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint32_t)));
    launch_and_copy_back_uint32(d_res, result, xor_kernel_u32, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_not_b32(uint32_t a, uint32_t* result) {
    uint32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint32_t)));
    launch_and_copy_back_uint32(d_res, result, not_kernel_u32, a);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_shl_b32(uint32_t a, uint32_t b, uint32_t* result) {
    uint32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint32_t)));
    launch_and_copy_back_uint32(d_res, result, shl_kernel_u32, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_shr_b32(uint32_t a, uint32_t b, uint32_t* result) {
    uint32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint32_t)));
    launch_and_copy_back_uint32(d_res, result, shr_kernel_u32, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

// Host wrappers for uint64 operations
void test_ptx_and_b64(uint64_t a, uint64_t b, uint64_t* result) {
    uint64_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint64_t)));
    launch_and_copy_back_uint64(d_res, result, and_kernel_u64, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_or_b64(uint64_t a, uint64_t b, uint64_t* result) {
    uint64_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint64_t)));
    launch_and_copy_back_uint64(d_res, result, or_kernel_u64, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_xor_b64(uint64_t a, uint64_t b, uint64_t* result) {
    uint64_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint64_t)));
    launch_and_copy_back_uint64(d_res, result, xor_kernel_u64, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_not_b64(uint64_t a, uint64_t* result) {
    uint64_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint64_t)));
    launch_and_copy_back_uint64(d_res, result, not_kernel_u64, a);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_shl_b64(uint64_t a, uint64_t b, uint64_t* result) {
    uint64_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint64_t)));
    launch_and_copy_back_uint64(d_res, result, shl_kernel_u64, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_shr_b64(uint64_t a, uint64_t b, uint64_t* result) {
    uint64_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint64_t)));
    launch_and_copy_back_uint64(d_res, result, shr_kernel_u64, a, b);
    CUDA_CHECK(cudaFree(d_res));
}