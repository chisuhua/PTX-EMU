/**
 * E2E co-simulation test: CUDA vectorAdd via CppTLM bridge path.
 *
 * This is a STANDARD CUDA program — no PTX-EMU-specific APIs.
 * When BUILD_LIB_CPPTLM_CUDART=ON, the environment auto-attaches
 * StubBridge and auto-advances via cudaDeviceSynchronize.
 *
 * Design: see openspec/changes/auto-co-sim-standalone/design.md
 * Spec:   see openspec/changes/auto-co-sim-standalone/specs/auto-co-simulation/spec.md
 */

#include "catch_amalgamated.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define VEC_SIZE 64

__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

TEST_CASE("cosim e2e: vectorAdd via bridge path",
          "[e2e][cosim][vector_add]") {
    float* h_A = (float*)malloc(VEC_SIZE * sizeof(float));
    float* h_B = (float*)malloc(VEC_SIZE * sizeof(float));
    float* h_C = (float*)malloc(VEC_SIZE * sizeof(float));
    float* golden = (float*)malloc(VEC_SIZE * sizeof(float));
    REQUIRE(h_A != nullptr);
    REQUIRE(h_B != nullptr);
    REQUIRE(h_C != nullptr);
    REQUIRE(golden != nullptr);

    for (int i = 0; i < VEC_SIZE; i++) {
        h_A[i] = (float)(i + 1);
        h_B[i] = (float)(i * 2);
        golden[i] = h_A[i] + h_B[i];
        h_C[i] = 0.0f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    REQUIRE(cudaMalloc(&d_A, VEC_SIZE * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_B, VEC_SIZE * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_C, VEC_SIZE * sizeof(float)) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_A, h_A, VEC_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_B, h_B, VEC_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // Standard CUDA: just launch and sync — no bridge attach, no advance().
    vectorAdd<<<1, VEC_SIZE>>>(d_A, d_B, d_C, VEC_SIZE);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    REQUIRE(cudaMemcpy(h_C, d_C, VEC_SIZE * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    int mismatches = 0;
    int first_bad = -1;
    float first_expected = 0.0f;
    float first_got = 0.0f;
    for (int i = 0; i < VEC_SIZE; i++) {
        if (h_C[i] != golden[i]) {
            if (mismatches == 0) {
                first_bad = i;
                first_expected = golden[i];
                first_got = h_C[i];
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("PASS: vectorAdd co-sim e2e — all %d elements match golden\n",
               VEC_SIZE);
    } else {
        printf("FAIL: %d/%d mismatches; first at i=%d expected=%f got=%f\n",
               mismatches, VEC_SIZE, first_bad, first_expected, first_got);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(golden);

    REQUIRE(mismatches == 0);
}