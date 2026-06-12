/**
 * Type 3 E2E test for ld.global hang regression (issue: fix-ldglobal-active-count-hang).
 *
 * Reproduces the minimal kernel pattern from bench/dummy-ldglobal/dummy-ldglobal.cu:
 *   block=64 (= 2 full warps, NO phantom lanes)
 *   + single ld.global u32 (in[i])
 *   + single st.global u32 (out[i] = in[i] + 1)
 *
 * Red phase expectation (before Fix #1): simulator HANGS at PC=9 because
 * sm_context.cpp:182-197 decrements is_active directly without syncing
 * WarpContext::active_count, so is_active() returns true forever → scheduler
 * never finishes the warp → cudaDeviceSynchronize() never returns → ctest
 * kills the process by timeout.
 *
 * Green phase expectation (after Fix #1 in sm_context.cpp + update_active_mask):
 * Test completes in < 5 seconds, all 64 elements match expected (i + 101).
 *
 * NOTE: kernel is intentionally NON-template. The CUDA driver requires concrete
 * types in __global__ functions; templates are used in bench/dummy-ldglobal only
 * because that driver is hand-rolled. Here we use plain int to keep the E2E
 * test focused on the ld.global bug, not template instantiation issues.
 */

#include "catch_amalgamated.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define SIZE 64

// Minimal kernel: one ld.global + one st.global, no shared, no barrier.
// Block = 64 threads => exactly 2 full warps, no phantom lanes.
__global__ void ldglobal_simple(int *in, int *out) {
    int i = threadIdx.x;
    int v = in[i];     // ld.global.u32 — triggers the blocked-decrement path
    out[i] = v + 1;    // st.global.u32
}

TEST_CASE("CUDA kernel: ld.global + st.global simple (hang regression)",
          "[e2e][ldglobal][hang_regression]") {
    int *in_h = (int *)malloc(SIZE * sizeof(int));
    int *out_h = (int *)malloc(SIZE * sizeof(int));
    REQUIRE(in_h != nullptr);
    REQUIRE(out_h != nullptr);

    for (int i = 0; i < SIZE; i++) {
        in_h[i] = i + 100;
        out_h[i] = 0;
    }

    int *in_d = nullptr;
    int *out_d = nullptr;
    cudaError_t err;

    err = cudaMalloc(&in_d, SIZE * sizeof(int));
    REQUIRE(err == cudaSuccess);
    err = cudaMalloc(&out_d, SIZE * sizeof(int));
    REQUIRE(err == cudaSuccess);

    err = cudaMemcpy(in_d, in_h, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    REQUIRE(err == cudaSuccess);

    // <<<1, 64>>> = exactly 2 warps, NO phantom lanes (control variable)
    ldglobal_simple<<<1, SIZE>>>(in_d, out_d);

    // If the simulator hangs on the ld.global, this call never returns and
    // ctest kills the process by timeout — that's the Red-phase signal.
    err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    err = cudaMemcpy(out_h, out_d, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);

    int mismatches = 0;
    int first_bad = -1;
    int first_expected = 0;
    int first_got = 0;
    for (int i = 0; i < SIZE; i++) {
        int expected = i + 100 + 1;
        if (out_h[i] != expected) {
            if (mismatches == 0) {
                first_bad = i;
                first_expected = expected;
                first_got = out_h[i];
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("PASS\n");
        INFO("ld.global simple kernel: all 64 elements match expected (i+101)");
    } else {
        printf("FAIL: %d/%d mismatches; first at i=%d expected=%d got=%d\n",
               mismatches, SIZE, first_bad, first_expected, first_got);
    }

    cudaFree(in_d);
    cudaFree(out_d);
    free(in_h);
    free(out_h);

    REQUIRE(mismatches == 0);
}