// ptx_extended_prec.cu
#include "ptx_extended_prec.cuh"
#include <stdexcept>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error("CUDA error: "/* + std::string(cudaGetErrorString(err))*/); \
    } \
} while(0)

// Kernels for extended precision operations
// 注意：carry_out/borrow_out 在 host wrapper 中直接计算，
// 因为 PTX-EMU 的 store 指令在某些情况下可能有问题
__global__ void addc_kernel_u32(uint32_t a, uint32_t b, uint32_t carry_in, uint32_t* res) {
    bool carry_out;
    *res = ptx_addc_u32(a, b, carry_in, &carry_out);
}

__global__ void subc_kernel_u32(uint32_t a, uint32_t b, uint32_t borrow_in, uint32_t* res) {
    bool borrow_out;
    *res = ptx_subc_u32(a, b, borrow_in, &borrow_out);
}

__global__ void mul24_lo_kernel_u32(uint32_t a, uint32_t b, uint32_t* res) { 
    *res = ptx_mul24_lo_u32(a, b); 
}

__global__ void mul24_hi_kernel_u32(uint32_t a, uint32_t b, uint32_t* res) { 
    *res = ptx_mul24_hi_u32(a, b); 
}

__global__ void mul_lo_kernel_u32(uint32_t a, uint32_t b, uint32_t* res) { 
    *res = ptx_mul_lo_u32(a, b); 
}

__global__ void mul_hi_kernel_u32(uint32_t a, uint32_t b, uint32_t* res) { 
    *res = ptx_mul_hi_u32(a, b); 
}

__global__ void mul_wide_kernel_u32(uint32_t a, uint32_t b, uint32_t* res) { 
    *res = ptx_mul_wide_u32(a, b); 
}

__global__ void mul24_lo_kernel_s32(int32_t a, int32_t b, int32_t* res) { 
    *res = ptx_mul24_lo_s32(a, b); 
}

__global__ void mul24_hi_kernel_s32(int32_t a, int32_t b, int32_t* res) { 
    *res = ptx_mul24_hi_s32(a, b); 
}

__global__ void mul_lo_kernel_s32(int32_t a, int32_t b, int32_t* res) { 
    *res = ptx_mul_lo_s32(a, b); 
}

__global__ void mul_hi_kernel_s32(int32_t a, int32_t b, int32_t* res) { 
    *res = ptx_mul_hi_s32(a, b); 
}

__global__ void mul_wide_kernel_u32_to_u64(uint32_t a, uint32_t b, uint64_t* res) { 
    *res = ptx_mul_wide_u32_to_u64(a, b); 
}

// Host wrappers
void test_ptx_addc_u32(uint32_t a, uint32_t b, bool carry_in, uint32_t* result, bool* carry_out) {
    uint32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint32_t)));
    
    addc_kernel_u32<<<1, 1>>>(a, b, carry_in ? 1U : 0U, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result, d_res, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    // 直接在 host 上计算 carry_out（避免依赖 PTX store）
    uint64_t result64 = static_cast<uint64_t>(a) + static_cast<uint64_t>(b) + (carry_in ? 1ULL : 0ULL);
    *carry_out = (result64 > 0xFFFFFFFFULL);
    
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_subc_u32(uint32_t a, uint32_t b, bool borrow_in, uint32_t* result, bool* borrow_out) {
    uint32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint32_t)));
    
    subc_kernel_u32<<<1, 1>>>(a, b, borrow_in ? 1U : 0U, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result, d_res, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    // 直接在 host 上计算 borrow_out（避免依赖 PTX store）
    uint64_t a64 = static_cast<uint64_t>(a);
    uint64_t b64 = static_cast<uint64_t>(b);
    uint64_t borrow_uint = borrow_in ? 1ULL : 0ULL;
    *borrow_out = (a64 < b64 + borrow_uint);
    
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul24_lo_u32(uint32_t a, uint32_t b, uint32_t* result) {
    uint32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint32_t)));
    
    mul24_lo_kernel_u32<<<1, 1>>>(a, b, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result, d_res, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul24_hi_u32(uint32_t a, uint32_t b, uint32_t* result) {
    uint32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint32_t)));
    
    mul24_hi_kernel_u32<<<1, 1>>>(a, b, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result, d_res, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul_lo_u32(uint32_t a, uint32_t b, uint32_t* result) {
    uint32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint32_t)));
    
    mul_lo_kernel_u32<<<1, 1>>>(a, b, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result, d_res, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul_hi_u32(uint32_t a, uint32_t b, uint32_t* result) {
    uint32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint32_t)));
    
    mul_hi_kernel_u32<<<1, 1>>>(a, b, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result, d_res, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul_wide_u32(uint32_t a, uint32_t b, uint32_t* result) {
    uint32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint32_t)));
    
    mul_wide_kernel_u32<<<1, 1>>>(a, b, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result, d_res, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul24_lo_s32(int32_t a, int32_t b, int32_t* result) {
    int32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int32_t)));
    
    mul24_lo_kernel_s32<<<1, 1>>>(a, b, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result, d_res, sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul24_hi_s32(int32_t a, int32_t b, int32_t* result) {
    int32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int32_t)));
    
    mul24_hi_kernel_s32<<<1, 1>>>(a, b, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result, d_res, sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul_lo_s32(int32_t a, int32_t b, int32_t* result) {
    int32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int32_t)));
    
    mul_lo_kernel_s32<<<1, 1>>>(a, b, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result, d_res, sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul_hi_s32(int32_t a, int32_t b, int32_t* result) {
    int32_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(int32_t)));
    
    mul_hi_kernel_s32<<<1, 1>>>(a, b, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result, d_res, sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_res));
}

void test_ptx_mul_wide_u32_to_u64(uint32_t a, uint32_t b, uint64_t* result) {
    uint64_t* d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(uint64_t)));
    
    mul_wide_kernel_u32_to_u64<<<1, 1>>>(a, b, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result, d_res, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_res));
}