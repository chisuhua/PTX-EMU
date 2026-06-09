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
    //__shared__ T shared_data[32];

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;

    int value;
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

    __syncthreads();
    output[tid] = value;

    /*
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
        output[tid] = shared_data[tid];
    }
    */
}

bool test_divergence_sync() {
    printf("test_divergence_sync: ");
    int *d_output;
    cudaMalloc(&d_output, 32 * sizeof(int));

    test_divergence_sync_kernel<int><<<1, 32>>>(d_output);

    int h_output[32];
    cudaMemcpy(h_output, d_output, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    // Verify each thread's value is correct
    bool failed = false;
    for (int i = 0; i < 32; i++) {
        int lane = i % WARP_SIZE;
        int expected;
        if (lane < 16) {
            // Path A: compute sum 0..lane
            expected = 0;
            for (int j = 0; j <= lane; j++) expected += j;
        } else {
            // Path B: compute product 1..(lane-15)
            expected = 1;
            for (int j = 1; j <= lane - 15; j++) expected *= j;
        }
        if (h_output[i] != expected) {
            printf("FAIL: h_output[%d] expected %d, got %d\n", i, expected, h_output[i]);
            failed = true;
        }
    }

    if (failed) {
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
