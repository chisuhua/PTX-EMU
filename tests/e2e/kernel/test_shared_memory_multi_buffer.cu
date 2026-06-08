#include "catch_amalgamated.hpp"
#include "ptxsim/execution_types.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/gpu_context.h"
#include <cuda.h>
#include <iostream>

using namespace ptxsim;

__global__ void kernel_multi_buffer(int* output_a, int* output_b, int num_threads) {
    __shared__ int buf_a[32];
    __shared__ int buf_b[32];
    
    int tid = threadIdx.x;
    
    // Write to both buffers without divergence
    buf_a[tid] = tid + 100;  // Offset by 100
    buf_b[tid] = tid + 200;  // Offset by 200
    
    // Barrier synchronization
    __syncthreads();
    
    // All threads read and verify cross-buffer visibility
    int val_a = buf_a[tid];
    int val_b = buf_b[tid];
    
    // Verify values match expected
    if (val_a == tid + 100 && val_b == tid + 200) {
        // Thread 0 writes verification result
        if (tid == 0) {
            output_a[0] = 1;  // Success flag
            output_b[0] = num_threads;  // Number of threads verified
        }
    }
}

TEST_CASE("CUDA kernel multi-buffer shared memory", "[e2e][shared_memory][multi_buffer]") {
    int* d_output_a = nullptr;
    int* d_output_b = nullptr;
    int size = sizeof(int);
    
    cudaError_t err = cudaMalloc(&d_output_a, size);
    REQUIRE(err == cudaSuccess);
    
    err = cudaMalloc(&d_output_b, size);
    REQUIRE(err == cudaSuccess);
    
    err = cudaMemset(d_output_a, 0, size);
    REQUIRE(err == cudaSuccess);
    
    err = cudaMemset(d_output_b, 0, size);
    REQUIRE(err == cudaSuccess);
    
    kernel_multi_buffer<<<1, 32>>>(d_output_a, d_output_b, 32);
    cudaDeviceSynchronize();
    
    int h_output_a = 0;
    int h_output_b = 0;
    
    err = cudaMemcpy(&h_output_a, d_output_a, size, cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);
    
    err = cudaMemcpy(&h_output_b, d_output_b, size, cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);
    
    INFO("Multi-buffer verification: flag=" << h_output_a << ", threads=" << h_output_b);
    REQUIRE(h_output_a == 1);  // Success flag
    REQUIRE(h_output_b == 32);  // All threads verified
    
    cudaFree(d_output_a);
    cudaFree(d_output_b);
}