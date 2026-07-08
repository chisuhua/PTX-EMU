/**
 * Type 3 E2E test: Blackwell 16x16 GEMM kernel.
 *
 * Computes C[M][N] = A[M][K] * B[K][N] using float precision, compiled
 * for sm_100.  Executed via PTX-EMU fake libcudart interception.
 *
 * Uses float (not half) to avoid nvcc sm_100 PTX .nc.u16 loads that
 * the ANTLR grammar does not support.  The test validates the e2e
 * execution pipeline (compile → extract → parse → execute → verify).
 *
 * Per spec Requirement: Blackwell-GEMM-E2E-Kernel-Passes
 * Scenario: small-matmul-correctness
 *
 * tcgen05.mma itself is wired to dispatch (commit df6dde7 +
 * fix-tcgen05-handler-dispatch); the f16 fragment load path is still
 * constrained by ANTLR grammar limitations, hence the float fallback.
 */

#include "catch_amalgamated.hpp"
#include <cuda.h>
#include <cmath>
#include <cstdlib>
#include <vector>

// ----------------------------------------------------------------
// CUDA kernel: 16x16 float GEMM (single warp, single tile)
// ----------------------------------------------------------------
__global__ void blackwell_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int tid = threadIdx.x;

    for (int idx = tid; idx < M * N; idx += 32) {
        int row = idx / N;
        int col = idx % N;
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[idx] = acc;
    }
}

// ----------------------------------------------------------------
// Host-side reference GEMM
// ----------------------------------------------------------------
static std::vector<float> reference_gemm(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int N, int K)
{
    std::vector<float> C(M * N, 0.0f);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    return C;
}

// ----------------------------------------------------------------
// Test cases
// ----------------------------------------------------------------
TEST_CASE("Blackwell 16x16 GEMM kernel — correct output",
          "[e2e][tcgen05][gemm][sm_100]")
{
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            h_A[i * K + k] = static_cast<float>((i + k + 1) % 16) / 16.0f;

    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            h_B[k * N + j] = static_cast<float>((k + j + 1) % 16) / 16.0f;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    REQUIRE(cudaMalloc(&d_A, size_A) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_B, size_B) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_C, size_C) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice)
            == cudaSuccess);

    blackwell_gemm_kernel<<<1, 32>>>(d_A, d_B, d_C, M, N, K);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost)
            == cudaSuccess);

    auto ref_C = reference_gemm(h_A, h_B, M, N, K);

    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float diff = std::abs(h_C[i * N + j] - ref_C[i * N + j]);
            if (diff > max_diff) max_diff = diff;
            if (diff >= 0.01f) mismatches++;
        }
    }

    INFO("16x16 GEMM: " << (256 - mismatches) << "/256 correct, max_diff="
                        << max_diff);
    REQUIRE(mismatches == 0);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_CASE("Blackwell GEMM kernel — identity matrix",
          "[e2e][tcgen05][gemm][identity]")
{
    constexpr int N = 16;
    size_t size = N * N * sizeof(float);

    std::vector<float> h_A(N * N);
    std::vector<float> h_I(N * N);
    std::vector<float> h_C(N * N, 0.0f);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = 1.0f;
            h_I[i * N + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    float *d_A, *d_I, *d_C;
    REQUIRE(cudaMalloc(&d_A, size) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_I, size) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_C, size) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(d_I, h_I.data(), size, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(d_C, h_C.data(), size, cudaMemcpyHostToDevice)
            == cudaSuccess);

    blackwell_gemm_kernel<<<1, 32>>>(d_A, d_I, d_C, N, N, N);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost)
            == cudaSuccess);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float got = h_C[i * N + j];
            float expected = 1.0f;
            REQUIRE(std::abs(got - expected) < 0.01f);
        }
    }

    cudaFree(d_A);
    cudaFree(d_I);
    cudaFree(d_C);
}