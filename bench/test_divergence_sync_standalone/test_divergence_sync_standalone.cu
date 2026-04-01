// ============================================================================
// Standalone test for divergence with __syncthreads()
// Isolated from bench/test_shared_memory/test_shared_memory.cu to diagnose
// barrier sync hanging issues
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define WARP_SIZE 32

// From test_shared_memory.cu: test_divergence_sync_kernel
template<typename T>
__global__ void test_divergence_sync_kernel(T *output) {
    __shared__ T shared_data[32];

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;

    T value;
    // Divergent paths: lane 0-15 vs 16-31
    if (lane < 16) {
        // Path A: compute sum 0..lane
        value = 0;
        for (int i = 0; i <= lane; i++) value += i;
    } else {
        // Path B: compute product 1..(lane-15)
        value = 1;
        for (int i = 1; i <= lane - 15; i++) value *= i;
    }

    shared_data[lane] = value;
    __syncthreads();

    // Thread 0 collects sum
    if (tid == 0) {
        T sum = 0;
        for (int i = 0; i < WARP_SIZE; i++) {
            sum += shared_data[i];
        }
        output[0] = sum;
    } else {
        output[tid] = 0;
    }
}

bool test_divergence_sync() {
    printf("test_divergence_sync: ");
    int *d_output;
    cudaMalloc(&d_output, 32 * sizeof(int));

    test_divergence_sync_kernel<int><<<1, 32>>>(d_output);

    int h_output[32];
    cudaMemcpy(h_output, d_output, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    // Compute expected sum
    int expected_sum = 0;
    for (int i = 0; i < 16; i++) {
        expected_sum += (i * (i + 1)) / 2;  // sum 0..i
    }
    for (int i = 16; i < 32; i++) {
        int prod = 1;
        for (int j = 1; j <= i - 15; j++) prod *= j;
        expected_sum += prod;
    }

    if (h_output[0] != expected_sum) {
        printf("FAIL: expected %d, got %d\n", expected_sum, h_output[0]);
        return false;
    }
    printf("PASS\n");
    return true;
}

int main() {
    printf("=== Standalone Divergence Sync Test ===\n\n");
    bool pass = test_divergence_sync();
    printf("\n=== Result: %s ===\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
