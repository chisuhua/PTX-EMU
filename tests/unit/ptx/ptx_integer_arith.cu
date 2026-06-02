// ptx_ops.cu
#include "ptx_integer_arith.cuh"
#include <stdexcept>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error("CUDA error: "/* + std::string(cudaGetErrorString(err))*/); \
    } \
} while(0)

template<typename Kernel, typename... Args>
void launch_and_copy_back(int* d_result, int* h_result, Kernel kernel, Args... args) {
    kernel<<<1, 1>>>(args..., d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
}

// Kernels
__global__ void add_kernel(int a, int b, int* res) { *res = ptx_add_s32(a, b); }
__global__ void sub_kernel(int a, int b, int* res) { *res = ptx_sub_s32(a, b); }
__global__ void mul_kernel(int a, int b, int* res) { *res = ptx_mul_s32(a, b); }
__global__ void mul24_kernel(int a, int b, int* res) { *res = ptx_mul24_s32(a, b); }
__global__ void mad_kernel(int a, int b, int c, int* res) { *res = ptx_mad_s32(a, b, c); }
__global__ void mad24_kernel(int a, int b, int c, int* res) { *res = ptx_mad24_s32(a, b, c); }
__global__ void div_kernel(int a, int b, int* res) { *res = ptx_div_s32(a, b); }
__global__ void rem_kernel(int a, int b, int* res) { *res = ptx_rem_s32(a, b); }
__global__ void abs_kernel(int a, int* res) { *res = ptx_abs_s32(a); }
__global__ void neg_kernel(int a, int* res) { *res = ptx_neg_s32(a); }
__global__ void min_kernel(int a, int b, int* res) { *res = ptx_min_s32(a, b); }
__global__ void max_kernel(int a, int b, int* res) { *res = ptx_max_s32(a, b); }
__global__ void popc_kernel(unsigned int a, int* res) { *res = ptx_popc_u32(a); }
__global__ void clz_kernel(unsigned int a, int* res) { *res = ptx_clz_u32(a); }
// __global__ void bfind_kernel(int a, int* res) { *res = ptx_bfind_s32(a); }

// Host wrappers
void test_ptx_add_s32(int a, int b, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, add_kernel, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_sub_s32(int a, int b, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, sub_kernel, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul_s32(int a, int b, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, mul_kernel, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul24_s32(int a, int b, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, mul24_kernel, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mad_s32(int a, int b, int c, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, mad_kernel, a, b, c);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mad24_s32(int a, int b, int c, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, mad24_kernel, a, b, c);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_div_s32(int a, int b, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, div_kernel, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_rem_s32(int a, int b, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, rem_kernel, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_abs_s32(int a, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, abs_kernel, a);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_neg_s32(int a, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, neg_kernel, a);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_min_s32(int a, int b, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, min_kernel, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_max_s32(int a, int b, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, max_kernel, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_popc_u32(unsigned int a, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, popc_kernel, a);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_clz_u32(unsigned int a, int* result) {
    int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    launch_and_copy_back(d_res, result, clz_kernel, a);
    CUDA_CHECK(cudaFree(d_res));
}

// void test_ptx_bfind_s32(int a, int* result) {
//     int* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
//     launch_and_copy_back(d_res, result, bfind_kernel, a);
//     CUDA_CHECK(cudaFree(d_res));
// }