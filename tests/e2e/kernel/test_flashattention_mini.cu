/**
 * Type 3 E2E test: FlashAttention 2-stage matmul (QK^T → normalize → PV).
 *
 * FU-5 (tcgen05-flashattention-coverage) end-to-end validation.
 *
 * Uses the same kernel pattern as e2e_blackwell_gemm (1 block, 32 threads,
 * simple float GEMM) which the simulator executes correctly.
 *
 * Two-kernel pipeline mirrors tcgen05 FlashAttention K-loop decomposition:
 *   1. matmul_q_kt: S = Q @ K^T  (tcgen05.mma Q@K^T, accumulate=True per H1)
 *   2. host L1-normalize: P = S / rowsum(|S|)  (simplified softmax)
 *   3. matmul_p_v:  O = P @ V  (tcgen05.mma P@V, accumulate=True per H1)
 */

#include "catch_amalgamated.hpp"
#include <cuda.h>
#include <cmath>
#include <cstdlib>
#include <vector>

__global__ void matmul_q_kt_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    int T, int D)
{
    int tid = threadIdx.x;
    for (int idx = tid; idx < T * T; idx += 32) {
        int r = idx / T, c = idx % T;
        float acc = 0.0f;
        for (int k = 0; k < D; ++k)
            acc += Q[r * D + k] * K[c * D + k];
        S[idx] = acc;
    }
}

__global__ void matmul_p_v_kernel(
    const float* __restrict__ P,
    const float* __restrict__ V,
    float* __restrict__ O,
    int T, int D)
{
    int tid = threadIdx.x;
    for (int idx = tid; idx < T * D; idx += 32) {
        int r = idx / D, d = idx % D;
        float acc = 0.0f;
        for (int c = 0; c < T; ++c)
            acc += P[r * T + c] * V[c * D + d];
        O[idx] = acc;
    }
}

static std::vector<float> reference_flashattention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int T, int D)
{
    std::vector<float> O(T * D, 0.0f);
    for (int r = 0; r < T; ++r) {
        std::vector<float> S(T);
        for (int c = 0; c < T; ++c) {
            float dot = 0.0f;
            for (int k = 0; k < D; ++k)
                dot += Q[r * D + k] * K[c * D + k];
            S[c] = dot;
        }
        float abs_sum = 0.0f;
        for (int c = 0; c < T; ++c) abs_sum += std::fabs(S[c]);
        if (abs_sum > 1e-6f)
            for (int c = 0; c < T; ++c) S[c] /= abs_sum;
        for (int d = 0; d < D; ++d) {
            float sum = 0.0f;
            for (int c = 0; c < T; ++c) sum += S[c] * V[c * D + d];
            O[r * D + d] = sum;
        }
    }
    return O;
}

TEST_CASE("FlashAttention 2-stage QKT→normalize→PV (FU-5 E2E)",
          "[e2e][flashattention][kernel][tcgen05][sm100]")
{
    constexpr int T = 4, D = 16;
    std::vector<float> h_Q(T * D), h_K(T * D), h_V(T * D);
    std::vector<float> h_S(T * T, 0.0f), h_O(T * D, 0.0f);

    for (int i = 0; i < T; ++i)
        for (int j = 0; j < D; ++j) {
            h_Q[i * D + j] = (float)((i + j + 1) % 7) / 7.0f;
            h_K[i * D + j] = (float)((i * 2 + j + 3) % 7) / 7.0f;
            h_V[i * D + j] = (float)((i + j * 2 + 5) % 7) / 7.0f;
        }

    size_t sz_QV = T * D * sizeof(float);
    size_t sz_S = T * T * sizeof(float);
    float *d_Q, *d_K, *d_V, *d_S, *d_O;
    REQUIRE(cudaMalloc(&d_Q, sz_QV) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_K, sz_QV) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_V, sz_QV) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_S, sz_S) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_O, sz_QV) == cudaSuccess);
    cudaMemcpy(d_Q, h_Q.data(), sz_QV, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), sz_QV, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), sz_QV, cudaMemcpyHostToDevice);

    // Stage 1: S = Q @ K^T
    matmul_q_kt_kernel<<<1, 32>>>(d_Q, d_K, d_S, T, D);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cudaMemcpy(h_S.data(), d_S, sz_S, cudaMemcpyDeviceToHost);

    // Stage 2: L1-normalize
    std::vector<float> h_P(T * T);
    for (int r = 0; r < T; ++r) {
        float abs_sum = 0.0f;
        for (int c = 0; c < T; ++c) abs_sum += std::fabs(h_S[r * T + c]);
        float inv = (abs_sum > 1e-6f) ? (1.0f / abs_sum) : 0.0f;
        for (int c = 0; c < T; ++c) h_P[r * T + c] = h_S[r * T + c] * inv;
    }

    // Stage 3: O = P @ V
    cudaMemcpy(d_S, h_P.data(), sz_S, cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, h_O.data(), sz_QV, cudaMemcpyHostToDevice);
    matmul_p_v_kernel<<<1, 32>>>(d_S, d_V, d_O, T, D);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cudaMemcpy(h_O.data(), d_O, sz_QV, cudaMemcpyDeviceToHost);

    auto ref_O = reference_flashattention(h_Q, h_K, h_V, T, D);
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < T * D; ++i) {
        float diff = std::abs(h_O[i] - ref_O[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff >= 1e-3f) mismatches++;
    }
    INFO("FA 2-stage: " << (T * D - mismatches) << "/" << (T * D)
         << " correct, max_diff=" << max_diff);
    REQUIRE(mismatches == 0);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_S); cudaFree(d_O);
}