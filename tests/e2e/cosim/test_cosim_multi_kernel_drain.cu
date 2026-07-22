//
// E2E test: multiple kernel drain via repeated cudaDeviceSynchronize
// (auto-co-sim spec)
//
// Launches N=3 kernels sequentially, each with an independent output
// buffer. Verifies all outputs are correct after synchronization.
// The bridge path should drain all pending kernels across multiple
// cudaDeviceSynchronize calls.
//
// Design: STANDARD CUDA program. No PTX-EMU-specific APIs.

#include "catch_amalgamated.hpp"
#include <cuda_runtime.h>
#include <cstdio>

#define NUM_KERNELS 3
#define VEC_SIZE 32

// Simple vectorAdd kernel
__global__ void vectorAddKernel(float* A, float* B, float* C, int N) {
    int i = threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

TEST_CASE("cosim e2e: multi-kernel drain with repeated sync",
          "[e2e][cosim][drain][multi]") {
    float h_A[VEC_SIZE], h_B[VEC_SIZE];
    float h_C[NUM_KERNELS][VEC_SIZE];
    float golden[VEC_SIZE];

    // Setup input data
    for (int i = 0; i < VEC_SIZE; i++) {
        h_A[i] = (float)(i + 1);
        h_B[i] = (float)(i * 2);
        golden[i] = h_A[i] + h_B[i];
    }

    // Allocate device memory (shared by all kernels)
    float *d_A, *d_B, *d_C[NUM_KERNELS];
    REQUIRE(cudaMalloc(&d_A, VEC_SIZE * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_B, VEC_SIZE * sizeof(float)) == cudaSuccess);
    for (int k = 0; k < NUM_KERNELS; k++) {
        REQUIRE(cudaMalloc(&d_C[k], VEC_SIZE * sizeof(float)) == cudaSuccess);
    }

    // Copy input data
    REQUIRE(cudaMemcpy(d_A, h_A, VEC_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_B, h_B, VEC_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // Launch NUM_KERNELS sequentially (same kernel, different output buffers)
    for (int k = 0; k < NUM_KERNELS; k++) {
        vectorAddKernel<<<1, VEC_SIZE>>>(d_A, d_B, d_C[k], VEC_SIZE);
    }

    // First sync — should drain at least one kernel
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    // Second sync — should drain remaining kernels without error
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    // Third sync — should be safe (no kernels left)
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    // Verify all output buffers have correct results
    for (int k = 0; k < NUM_KERNELS; k++) {
        REQUIRE(cudaMemcpy(h_C[k], d_C[k], VEC_SIZE * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess);

        int mismatches = 0;
        for (int i = 0; i < VEC_SIZE; i++) {
            if (h_C[k][i] != golden[i]) mismatches++;
        }

        if (mismatches == 0) {
            printf("PASS: kernel %d — all %d elements match golden\n",
                   k, VEC_SIZE);
        } else {
            printf("FAIL: kernel %d — %d mismatches\n", k, mismatches);
        }
        REQUIRE(mismatches == 0);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    for (int k = 0; k < NUM_KERNELS; k++) cudaFree(d_C[k]);

    SUCCEED("all 3 kernels drained correctly");
}