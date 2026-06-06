// ptx_float_arith.cu
#include "ptx_float_arith.cuh"
#include <stdexcept>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error("CUDA error: "/* + std::string(cudaGetErrorString(err))*/); \
    } \
} while(0)

template<typename Kernel, typename... Args>
void launch_and_copy_back(float* d_result, float* h_result, Kernel kernel, Args... args) {
    kernel<<<1, 1>>>(args..., d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
}

template<typename Kernel, typename... Args>
void launch_and_copy_back_double(double* d_result, double* h_result, Kernel kernel, Args... args) {
    kernel<<<1, 1>>>(args..., d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
}

// Kernels for float operations
__global__ void add_kernel_f32(float a, float b, float* res) { *res = ptx_add_f32(a, b); }
__global__ void sub_kernel_f32(float a, float b, float* res) { *res = ptx_sub_f32(a, b); }
__global__ void mul_kernel_f32(float a, float b, float* res) { *res = ptx_mul_f32(a, b); }
__global__ void div_kernel_f32(float a, float b, float* res) { *res = ptx_div_f32(a, b); }
__global__ void abs_kernel_f32(float a, float* res) { *res = ptx_abs_f32(a); }
__global__ void neg_kernel_f32(float a, float* res) { *res = ptx_neg_f32(a); }
__global__ void min_kernel_f32(float a, float b, float* res) { *res = ptx_min_f32(a, b); }
__global__ void max_kernel_f32(float a, float b, float* res) { *res = ptx_max_f32(a, b); }
__global__ void sqrt_kernel_f32(float a, float* res) { *res = ptx_sqrt_f32(a); }
__global__ void rcp_kernel_f32(float a, float* res) { *res = ptx_rcp_f32(a); }


// Kernels for double operations
__global__ void add_kernel_f64(double a, double b, double* res) { *res = ptx_add_f64(a, b); }
__global__ void sub_kernel_f64(double a, double b, double* res) { *res = ptx_sub_f64(a, b); }
__global__ void mul_kernel_f64(double a, double b, double* res) { *res = ptx_mul_f64(a, b); }
__global__ void div_kernel_f64(double a, double b, double* res) { *res = ptx_div_f64(a, b); }
__global__ void abs_kernel_f64(double a, double* res) { *res = ptx_abs_f64(a); }
__global__ void neg_kernel_f64(double a, double* res) { *res = ptx_neg_f64(a); }
__global__ void min_kernel_f64(double a, double b, double* res) { *res = ptx_min_f64(a, b); }
__global__ void max_kernel_f64(double a, double b, double* res) { *res = ptx_max_f64(a, b); }
__global__ void sqrt_kernel_f64(double a, double* res) { *res = ptx_sqrt_f64(a); }
__global__ void rcp_kernel_f64(double a, double* res) { *res = ptx_rcp_f64(a); }

// Host wrappers for float
void test_ptx_add_f32(float a, float b, float* result) {
    float* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));
    launch_and_copy_back(d_res, result, add_kernel_f32, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_sub_f32(float a, float b, float* result) {
    float* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));
    launch_and_copy_back(d_res, result, sub_kernel_f32, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul_f32(float a, float b, float* result) {
    float* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));
    launch_and_copy_back(d_res, result, mul_kernel_f32, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_div_f32(float a, float b, float* result) {
    float* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));
    launch_and_copy_back(d_res, result, div_kernel_f32, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_abs_f32(float a, float* result) {
    float* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));
    launch_and_copy_back(d_res, result, abs_kernel_f32, a);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_neg_f32(float a, float* result) {
    float* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));
    launch_and_copy_back(d_res, result, neg_kernel_f32, a);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_min_f32(float a, float b, float* result) {
    float* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));
    launch_and_copy_back(d_res, result, min_kernel_f32, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_max_f32(float a, float b, float* result) {
    float* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));
    launch_and_copy_back(d_res, result, max_kernel_f32, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_sqrt_f32(float a, float* result) {
    float* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));
    launch_and_copy_back(d_res, result, sqrt_kernel_f32, a);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_rcp_f32(float a, float* result) {
    float* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));
    launch_and_copy_back(d_res, result, rcp_kernel_f32, a);
    CUDA_CHECK(cudaFree(d_res));
}


// Host wrappers for double
void test_ptx_add_f64(double a, double b, double* result) {
    double* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(double)));
    launch_and_copy_back_double(d_res, result, add_kernel_f64, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_sub_f64(double a, double b, double* result) {
    double* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(double)));
    launch_and_copy_back_double(d_res, result, sub_kernel_f64, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul_f64(double a, double b, double* result) {
    double* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(double)));
    launch_and_copy_back_double(d_res, result, mul_kernel_f64, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_div_f64(double a, double b, double* result) {
    double* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(double)));
    launch_and_copy_back_double(d_res, result, div_kernel_f64, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_abs_f64(double a, double* result) {
    double* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(double)));
    launch_and_copy_back_double(d_res, result, abs_kernel_f64, a);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_neg_f64(double a, double* result) {
    double* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(double)));
    launch_and_copy_back_double(d_res, result, neg_kernel_f64, a);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_min_f64(double a, double b, double* result) {
    double* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(double)));
    launch_and_copy_back_double(d_res, result, min_kernel_f64, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_max_f64(double a, double b, double* result) {
    double* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(double)));
    launch_and_copy_back_double(d_res, result, max_kernel_f64, a, b);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_sqrt_f64(double a, double* result) {
    double* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(double)));
    launch_and_copy_back_double(d_res, result, sqrt_kernel_f64, a);
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_rcp_f64(double a, double* result) {
    double* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(double)));
    launch_and_copy_back_double(d_res, result, rcp_kernel_f64, a);
    CUDA_CHECK(cudaFree(d_res));
}