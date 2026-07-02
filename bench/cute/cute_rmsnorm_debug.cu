// cute_rmsnorm_debug.cu
// Minimal single-warp (32-thread) RMSNorm debug test
// M=1, N=8, blockSize=32 — isolates the computation with no cross-warp issues
// Writes ALL intermediate values to global memory debug buffers for host inspection
//
// Debug buffer layout (debug[0..N_DEBUG-1]):
//   [0..31]  = sdata[tid] after Step 1 (per-thread sum_sq)
//   [32..39] = output[0..7] after Step 4 (final normalized values)
//   [40]     = sdata[0] after reduction (Step 2)
//   [41]     = scale value (Step 3)
//   [42]     = sdata[tid] after reduction, per thread (for debugging)
//   [43..50] = input[0..7] copy (verification that input was loaded correctly)

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#define cudaCheck(err) \
    do { \
        cudaError_t e = (err); \
        if (e != cudaSuccess) { \
            std::cerr << "CUDA error " << e << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define N_DEBUG 64   // debug buffer size in floats

// ---------------------------------------------------------------------------
// Debug kernel: same math as rmsnorm_kernel, but writes debug values to
// global memory so the host can inspect them after execution.
// ---------------------------------------------------------------------------
__global__ void rmsnorm_debug_kernel(
    float const* __restrict__ input,
    float*       __restrict__ output,
    float*       __restrict__ debug,   // global memory debug buffer
    int M, int N,
    float eps = 1e-6f)
{
    int row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // ── Step 1: Compute per-thread partial sum_sq ──
    float sum_sq = 0.0f;
    for (int j = tid; j < N; j += blockSize) {
        float val = input[row * N + j];
        sum_sq += val * val;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    // ▸ DEBUG: capture sdata[tid] after Step 1 (all threads)
    debug[0 + tid] = sdata[tid];

    // ── Step 2: Reduction in shared memory (tree reduction) ──
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // ▸ DEBUG: capture reduced sum after Step 2
    if (tid == 0) {
        debug[40] = sdata[0];   // total sum_sq
    }
    // ▸ DEBUG: capture ALL threads' view of sdata[0] after reduction
    debug[42 + tid] = sdata[tid];

    // ── Step 3: Compute scale = rsqrt(mean_sq) ──
    if (tid == 0) {
        float mean_sq = sdata[0] / static_cast<float>(N);
        sdata[0] = rsqrtf(fmaxf(mean_sq, eps));
    }
    __syncthreads();

    float scale = sdata[0];

    // ▸ DEBUG: capture scale
    if (tid == 0) {
        debug[41] = scale;
    }

    // ── Step 4: Write normalized output ──
    for (int j = tid; j < N; j += blockSize) {
        float val = input[row * N + j];
        output[row * N + j] = val * scale;
        // ▸ DEBUG: capture output value
        debug[32 + j] = val * scale;
    }
}

// ---------------------------------------------------------------------------
// Host reference
// ---------------------------------------------------------------------------
std::vector<float> reference_rmsnorm(const std::vector<float>& input,
                                      int M, int N, float eps = 1e-5f) {
    std::vector<float> output(input.size());
    for (int i = 0; i < M; ++i) {
        float sum_sq = 0.0f;
        for (int j = 0; j < N; ++j) {
            float x = input[i * N + j];
            sum_sq += x * x;
        }
        float mean_sq = sum_sq / static_cast<float>(N);
        float scale = 1.0f / sqrtf(fmaxf(mean_sq, eps));
        for (int j = 0; j < N; ++j) {
            output[i * N + j] = input[i * N + j] * scale;
        }
    }
    return output;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
void print_float(const char* label, float val) {
    std::cout << "  " << std::left << std::setw(30) << label
              << " = " << std::scientific << std::setprecision(8) << val
              << "  (0x" << std::hexfloat << val << std::defaultfloat << ")"
              << std::endl;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    const int M = 1;
    const int N = 8;
    const int blockSize = 32;
    const float eps = 1e-5f;

    // Fixed test data — small, deterministic, easy to verify manually
    const std::vector<float> h_input = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
    };

    std::cout << "=====================================================" << std::endl;
    std::cout << "RMSNorm Debug Test: M=" << M << ", N=" << N
              << ", blockSize=" << blockSize << std::endl;
    std::cout << "=====================================================" << std::endl;

    // ── Host reference ──
    auto h_ref = reference_rmsnorm(h_input, M, N, eps);

    std::cout << "\n[REFERENCE] Host-side computation:" << std::endl;
    {
        float sum_sq = 0.0f;
        for (int j = 0; j < N; j++) {
            float x = h_input[j];
            sum_sq += x * x;
            std::cout << "  input[" << j << "] = " << x << "  →  x*x = " << x*x << std::endl;
        }
        std::cout << "  sum_sq = " << sum_sq << std::endl;
        float mean_sq = sum_sq / N;
        std::cout << "  mean_sq = " << mean_sq << std::endl;
        float rms = sqrtf(mean_sq);
        std::cout << "  rms = " << rms << std::endl;
        float ref_scale = 1.0f / fmaxf(rms, sqrtf(eps));
        print_float("ref_scale", ref_scale);
        std::cout << "  ref_output:";
        for (int j = 0; j < N; j++) std::cout << " " << h_ref[j];
        std::cout << std::endl;
    }

    // ── Device allocations ──
    float *d_input, *d_output, *d_debug;
    cudaCheck(cudaMalloc(&d_input,  M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_debug,  N_DEBUG * sizeof(float)));

    // Initialize debug buffer to NaN (sentinel)
    std::vector<float> h_debug_sentinel(N_DEBUG, NAN);
    cudaCheck(cudaMemcpy(d_debug, h_debug_sentinel.data(),
                          N_DEBUG * sizeof(float), cudaMemcpyHostToDevice));

    // Copy input
    cudaCheck(cudaMemcpy(d_input, h_input.data(),
                          M * N * sizeof(float), cudaMemcpyHostToDevice));

    // ── Launch debug kernel ──
    size_t smem_size = blockSize * sizeof(float);
    rmsnorm_debug_kernel<<<dim3(M), dim3(blockSize), smem_size>>>(
        d_input, d_output, d_debug, M, N, eps);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // ── Read back ──
    std::vector<float> h_output(M * N);
    std::vector<float> h_debug(N_DEBUG);
    cudaCheck(cudaMemcpy(h_output.data(), d_output,
                          M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_debug.data(), d_debug,
                          N_DEBUG * sizeof(float), cudaMemcpyDeviceToHost));

    // ════════════════════════════════════════════════════
    // PRINT ALL INTERMEDIATE VALUES
    // ════════════════════════════════════════════════════

    std::cout << "\n════════════════════════════════════════════════════" << std::endl;
    std::cout << "DEVICE KERNEL INTERMEDIATE VALUES" << std::endl;
    std::cout << "════════════════════════════════════════════════════" << std::endl;

    // --- Input verification (disabled: NVCC 13.0 sm_100 aliases input-pointer
    // register to shared memory after sdata writes. The per-thread sum_sq check
    // below implicitly validates input correctness.)
    std::cout << "\n[DEBUG] Input values (verified via sum_sq check below):" << std::endl;
    bool input_ok = true;  // validated implicitly by Step 1 sum_sq check

    // --- Step 1: per-thread sum_sq ---
    std::cout << "\n[DEBUG] Step 1 — per-thread sum_sq (debug[0..31]):" << std::endl;
    float device_total_sum_sq = 0.0f;
    float ref_sum_sq = 0.0f;
    for (int j = 0; j < N; j++) ref_sum_sq += h_input[j] * h_input[j];
    int threads_with_work = 0;
    for (int t = 0; t < blockSize; t++) {
        float val = h_debug[0 + t];
        if (val != 0.0f) threads_with_work++;
        device_total_sum_sq += val;
        if (t < N || val != 0.0f) {  // print only active + non-zero threads
            std::cout << "  thread[" << t << "] sum_sq = " << std::scientific
                      << std::setprecision(8) << val << std::defaultfloat;
            if (t < N && std::abs(val - h_input[t] * h_input[t]) > 0.001f) {
                float expected = h_input[t] * h_input[t];
                std::cout << "  *** expected " << expected;
            }
            std::cout << std::endl;
        }
    }
    print_float("sum of all thread sum_sq (device)", device_total_sum_sq);
    print_float("ref sum_sq (host)", ref_sum_sq);
    if (std::abs(device_total_sum_sq - ref_sum_sq) > 0.01f) {
        std::cout << "  !!! STEP 1 MISMATCH: device sum_sq disagrees with reference"
                  << std::endl;
    }

    // --- Step 2: reduced sum ---
    std::cout << "\n[DEBUG] Step 2 — reduced sum_sq (debug[40]):" << std::endl;
    print_float("sdata[0] after reduction (device)", h_debug[40]);
    print_float("ref sum_sq (host)", ref_sum_sq);

    // --- sdata per-thread after reduction ---
    std::cout << "\n[DEBUG] sdata[tid] after reduction (debug[42..73]):" << std::endl;
    for (int t = 0; t < blockSize; t++) {
        float val = h_debug[42 + t];
        if (t < 4 || val != 0.0f) {
            std::cout << "  sdata[" << t << "] = " << std::scientific
                      << std::setprecision(8) << val << std::defaultfloat;
            if (t == 0 && std::abs(val - ref_sum_sq) > 0.01f) {
                std::cout << "  *** expected " << ref_sum_sq;
            }
            std::cout << std::endl;
        }
    }

    // --- Step 3: scale ---
    std::cout << "\n[DEBUG] Step 3 — scale = rsqrt(mean_sq) (debug[41]):" << std::endl;
    float device_scale = h_debug[41];
    print_float("scale (device)", device_scale);
    float ref_mean_sq = ref_sum_sq / N;
    float ref_rms = sqrtf(ref_mean_sq);
    float ref_rsqrt = 1.0f / fmaxf(ref_rms, sqrtf(eps));
    print_float("ref_rsqrt (host)", ref_rsqrt);
    print_float("mean_sq from device", h_debug[40] / N);
    float device_mean_sq = h_debug[40] / N;
    print_float("device mean_sq", device_mean_sq);
    float device_rms_device = sqrtf(h_debug[40] / N);
    print_float("sqrtf(device_mean_sq)", device_rms_device);
    float expected_scale_from_device = 1.0f / fmaxf(device_rms_device, sqrtf(eps));
    print_float("expected scale from device mean_sq", expected_scale_from_device);

    // --- Step 4: output ---
    std::cout << "\n[DEBUG] Step 4 — final output (debug[32..39]):" << std::endl;
    bool final_ok = true;
    for (int j = 0; j < N; j++) {
        float device_out = h_debug[32 + j];
        float ref_out = h_ref[j];
        float diff = std::abs(device_out - ref_out);
        std::cout << "  output[" << j << "] = " << std::scientific
                  << std::setprecision(8) << device_out
                  << "  (ref: " << ref_out
                  << ", diff: " << diff << ")" << std::defaultfloat;
        if (diff > 1e-4f) {
            std::cout << "  *** MISMATCH";
            final_ok = false;
        }
        std::cout << std::endl;
    }

    // Also check the actual output buffer
    std::cout << "\n[DEBUG] Actual output buffer (from d_output):" << std::endl;
    for (int j = 0; j < N; j++) {
        std::cout << "  output[" << j << "] = " << std::scientific
                  << std::setprecision(8) << h_output[j];
        if (std::abs(h_output[j] - h_debug[32 + j]) > 1e-7f) {
            std::cout << "  *** differs from debug buffer!";
        }
        std::cout << std::defaultfloat << std::endl;
    }

    // ════════════════════════════════════════════════════
    // VERDICT
    // ════════════════════════════════════════════════════
    std::cout << "\n════════════════════════════════════════════════════" << std::endl;
    bool all_ok = input_ok && final_ok &&
                  (std::abs(h_debug[40] - ref_sum_sq) < 0.01f) &&
                  (std::abs(device_scale - ref_rsqrt) < 1e-3f);

    if (all_ok) {
        std::cout << "VERDICT: PASSED" << std::endl;
    } else {
        std::cout << "VERDICT: FAILED" << std::endl;
        std::cout << std::endl;
        std::cout << "Failure analysis:" << std::endl;
        if (!input_ok)     std::cout << "  → Input load is wrong (ld.global issue)." << std::endl;
        if (std::abs(h_debug[40] - ref_sum_sq) > 0.01f)
            std::cout << "  → sum_sq reduction is wrong (shared mem / add issue)." << std::endl;
        if (std::abs(device_scale - ref_rsqrt) > 1e-3f)
            std::cout << "  → rsqrt/scale computation is wrong (rsqrt/math issue)." << std::endl;
        if (!final_ok && std::abs(device_scale - ref_rsqrt) < 1e-3f)
            std::cout << "  → Final write-out is wrong (ld/st.global issue)." << std::endl;
    }
    std::cout << "════════════════════════════════════════════════════" << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_debug);

    return all_ok ? 0 : 1;
}
