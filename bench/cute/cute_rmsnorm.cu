// rmsnorm_cute.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

// CUTLASS CUTE headers (only needed for layout abstraction; we use minimal subset)
#include "cute/tensor.hpp"
using namespace cute;

// ----------------------------
// Host-side reference RMSNorm (for validation)
// ----------------------------
std::vector<float> reference_rmsnorm(const std::vector<float>& input, int M, int N, float eps = 1e-6f) {
    std::vector<float> output(input.size());
    for (int i = 0; i < M; ++i) {
        float sum_sq = 0.0f;
        for (int j = 0; j < N; ++j) {
            float x = input[i * N + j];
            sum_sq += x * x;
        }
        float rms = sqrtf(sum_sq / static_cast<float>(N));
        float scale = 1.0f / fmaxf(rms, sqrtf(eps)); // equivalent to rsqrt(mean_sq + eps)
        for (int j = 0; j < j < N; ++j) {
            output[i * N + j] = input[i * N + j] * scale;
        }
    }
    return output;
}

// ----------------------------
// CUDA Kernel (CUTE-style layout abstraction)
// ----------------------------
template <typename T>
__global__ void rmsnorm_kernel(
    T const* __restrict__ input,
    T*       __restrict__ output,
    int M, int N,
    float eps = 1e-6f)
{
    // Use CUTE to describe global layout (row-major [M, N])
    auto gLayout = make_layout(make_shape(M, N), make_stride(N, _1{}));

    int row = blockIdx.x;
    if (row >= M) return;

    // Extract the row as a rank-1 tensor
    auto input_row  = gLayout(row, _);
    auto output_row = gLayout(row, _);

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Step 1: Compute sum of squares
    float sum_sq = 0.0f;
    for (int j = tid; j < N; j += blockSize) {
        T val = input_row(j);
        sum_sq += static_cast<float>(val) * static_cast<float>(val);
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    // Step 2: Reduction in shared memory
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Step 3: Compute reciprocal RMS and scale
    if (tid == 0) {
        float mean_sq = sdata[0] / static_cast<float>(N);
        sdata[0] = rsqrtf(fmaxf(mean_sq, eps)); // safe rsqrt
    }
    __syncthreads();

    float scale = sdata[0];

    // Step 4: Write normalized output
    for (int j = tid; j < N; j += blockSize) {
        T val = input_row(j);
        output_row(j) = static_cast<T>(static_cast<float>(val) * scale);
    }
}

// ----------------------------
// Host launch wrapper
// ----------------------------
void launch_rmsnorm(float const* d_input, float* d_output, int M, int N, cudaStream_t stream = 0) {
    constexpr int kBlockSize = 256;
    dim3 grid(M);
    dim3 block(kBlockSize);
    size_t smem_size = kBlockSize * sizeof(float);

    rmsnorm_kernel<float><<<grid, block, smem_size, stream>>>(d_input, d_output, M, N);
    cudaCheck(cudaGetLastError());
}

// ----------------------------
// Utility
// ----------------------------
#define cudaCheck(err) \
    do { \
        cudaError_t e = (err); \
        if (e != cudaSuccess) { \
            std::cerr << "CUDA error " << e << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// ----------------------------
// Main test
// ----------------------------
int main() {
    const int M = 8;   // batch or sequence length
    const int N = 768; // hidden size
    const float eps = 1e-6f;

    std::cout << "Testing RMSNorm with M=" << M << ", N=" << N << std::endl;

    // Host input
    std::vector<float> h_input(M * N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (auto& x : h_input) x = dis(gen);

    // Reference output
    auto h_ref = reference_rmsnorm(h_input, M, N, eps);

    // Device allocation
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input,  M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, M * N * sizeof(float)));

    // Copy to device
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    launch_rmsnorm(d_input, d_output, M, N);

    // Copy back
    std::vector<float> h_output(M * N);
    cudaCheck(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Validate
    const float tol = 1e-5f;
    bool passed = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(h_output[i] - h_ref[i]) > tol) {
            std::cerr << "Mismatch at [" << i << "]: got " << h_output[i]
                      << ", expected " << h_ref[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "✅ RMSNorm test PASSED!" << std::endl;
    } else {
        std::cout << "❌ RMSNorm test FAILED!" << std::endl;
        return 1;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}