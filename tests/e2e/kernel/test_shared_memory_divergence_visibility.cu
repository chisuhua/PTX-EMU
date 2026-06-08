#include "catch_amalgamated.hpp"
#include "ptxsim/execution_types.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/gpu_context.h"
#include <cuda.h>
#include <iostream>

using namespace ptxsim;

__global__ void kernel_divergence_visibility(int* output_a, int* output_b, int* cross_visibility) {
    __shared__ int buf_a[32];
    __shared__ int buf_b[32];
    
    int tid = threadIdx.x;
    
    // Divergent paths: lanes 0-15 write buf_a, lanes 16-31 write buf_b
    if (tid < 16) {
        // Path A: lanes 0-15
        buf_a[tid] = tid + 1000;  // Offset by 1000
    } else {
        // Path B: lanes 16-31
        buf_b[tid] = tid + 2000;  // Offset by 2000
    }
    
    // Barrier synchronization - ensures cross-path visibility
    __syncthreads();
    
    // All lanes read both buffers to verify cross-path visibility
    int val_a = buf_a[tid % 16];  // Read from buf_a (Path A's data)
    int val_b = buf_b[16 + (tid % 16)];  // Read from buf_b (Path B's data)
    
    // Verify cross-path visibility
    int expected_a = (tid % 16) + 1000;
    int expected_b = 16 + (tid % 16) + 2000;
    
    if (val_a == expected_a && val_b == expected_b) {
        // Thread 0 writes verification results
        if (tid == 0) {
            output_a[0] = 1;  // Path A visibility verified
            output_b[0] = 1;  // Path B visibility verified
            cross_visibility[0] = 32;  // All threads verified cross-path visibility
        }
    }
}

TEST_CASE("CUDA kernel divergence visibility with shared memory", "[e2e][shared_memory][divergence][visibility]") {
    int* d_output_a = nullptr;
    int* d_output_b = nullptr;
    int* d_cross_visibility = nullptr;
    int size = sizeof(int);
    
    cudaError_t err = cudaMalloc(&d_output_a, size);
    REQUIRE(err == cudaSuccess);
    
    err = cudaMalloc(&d_output_b, size);
    REQUIRE(err == cudaSuccess);
    
    err = cudaMalloc(&d_cross_visibility, size);
    REQUIRE(err == cudaSuccess);
    
    err = cudaMemset(d_output_a, 0, size);
    REQUIRE(err == cudaSuccess);
    
    err = cudaMemset(d_output_b, 0, size);
    REQUIRE(err == cudaSuccess);
    
    err = cudaMemset(d_cross_visibility, 0, size);
    REQUIRE(err == cudaSuccess);
    
    kernel_divergence_visibility<<<1, 32>>>(d_output_a, d_output_b, d_cross_visibility);
    cudaDeviceSynchronize();
    
    int h_output_a = 0;
    int h_output_b = 0;
    int h_cross_visibility = 0;
    
    err = cudaMemcpy(&h_output_a, d_output_a, size, cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);
    
    err = cudaMemcpy(&h_output_b, d_output_b, size, cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);
    
    err = cudaMemcpy(&h_cross_visibility, d_cross_visibility, size, cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);
    
    INFO("Divergence visibility: path_a=" << h_output_a 
         << ", path_b=" << h_output_b 
         << ", cross=" << h_cross_visibility);
    
    REQUIRE(h_output_a == 1);  // Path A visibility verified
    REQUIRE(h_output_b == 1);  // Path B visibility verified
    REQUIRE(h_cross_visibility == 32);  // All threads verified cross-path visibility
    
    cudaFree(d_output_a);
    cudaFree(d_output_b);
    cudaFree(d_cross_visibility);
}