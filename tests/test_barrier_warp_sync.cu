#include "catch_amalgamated.hpp"
#include "ptxsim/execution_types.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/gpu_context.h"
#include <cuda.h>
#include <iostream>

using namespace ptxsim;

__device__ int d_output[32] = {0};
__device__ int d_reduced_sum = 0;

__global__ void kernel_barrier_sync(int* output, int num_threads) {
    __shared__ int shared_data[32];

    int tid = threadIdx.x;

    shared_data[tid] = tid + 1;

    __syncthreads();

    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < num_threads; i++) {
            sum += shared_data[i];
        }
        d_reduced_sum = sum;
        output[0] = sum;
    }
}

__global__ void kernel_atomic_barrier(int* counter_out) {
    __shared__ int counter;

    if (threadIdx.x == 0) {
        counter = 0;
    }
    __syncthreads();

    atomicAdd(&counter, 1);

    __syncthreads();

    if (threadIdx.x == 0) {
        *counter_out = counter;
    }
}

TEST_CASE("CUDA kernel barrier synchronization", "[barrier][cuda][e2e]") {
    int* d_output_ptr = nullptr;
    int size = sizeof(int) * 32;

    cudaError_t err = cudaMalloc(&d_output_ptr, size);
    REQUIRE(err == cudaSuccess);

    err = cudaMemset(d_output_ptr, 0, size);
    REQUIRE(err == cudaSuccess);

    kernel_barrier_sync<<<1, 32>>>(d_output_ptr, 32);
    cudaDeviceSynchronize();

    int h_output[32] = {0};
    err = cudaMemcpy(h_output, d_output_ptr, size, cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);

    int expected_sum = 0;
    for (int i = 0; i < 32; i++) {
        expected_sum += (i + 1);
    }
    REQUIRE(h_output[0] == expected_sum);

    cudaFree(d_output_ptr);
}

TEST_CASE("CUDA kernel atomic counter with barrier", "[barrier][cuda][e2e]") {
    int* d_counter_ptr = nullptr;
    cudaMalloc(&d_counter_ptr, sizeof(int));

    kernel_atomic_barrier<<<1, 32>>>(d_counter_ptr);
    cudaDeviceSynchronize();

    int h_counter = -1;
    cudaMemcpy(&h_counter, d_counter_ptr, sizeof(int), cudaMemcpyDeviceToHost);

    INFO("counter value: " << h_counter);
    REQUIRE(h_counter != -1);

    cudaFree(d_counter_ptr);
    cudaDeviceReset();
}