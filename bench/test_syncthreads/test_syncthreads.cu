#include <cstdio>
#include <cstdlib>

// Test 1: Basic shared memory write/read with barrier
template<typename T>
__global__ void test_shared_barrier(T *output) {
    __shared__ T shared_data[32];

    int tid = threadIdx.x;
    // Phase 1: Each thread writes its index to shared memory
    shared_data[tid] = tid;

    // Barrier: all threads must complete their writes before any can read
    __syncthreads();

    // Phase 2: Each thread reads from a different location
    // Thread i reads from location (31-i)
    T value = shared_data[31 - tid];

    // Another barrier to ensure all reads complete
    __syncthreads();

    // Write result
    output[tid] = value;
}

// Test 2: Multi-block barrier test (each block operates independently)
template<typename T>
__global__ void test_multi_block_barrier(T *output) {
    __shared__ T block_data[32];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Each block initializes its shared memory differently based on block ID
    block_data[tid] = bid * 100 + tid;

    __syncthreads();

    // Each thread computes sum of its block's data
    T sum = 0;
    for (int i = 0; i < 32; i++) {
        sum += block_data[i];
    }

    __syncthreads();

    output[bid * blockDim.x + tid] = sum;
}

// Test 3: Nested sync with computation
template<typename T>
__global__ void test_nested_sync(T *output) {
    __shared__ T data_a[16];
    __shared__ T data_b[16];

    int tid = threadIdx.x;

    // First phase: fill data_a
    data_a[tid] = tid;
    __syncthreads();

    // Copy data_a to data_b with offset
    if (tid < 16) {
        data_b[tid] = data_a[tid] + data_a[(tid + 1) % 16];
    }
    __syncthreads();

    // Write output
    output[tid] = data_b[tid];
}

template<typename T>
bool run_tests() {
    bool all_pass = true;
    T *d_output;

    // Test 1: Basic barrier
    {
        cudaMalloc(&d_output, 32 * sizeof(T));
        test_shared_barrier<T><<<1, 32>>>(d_output);

        T h_output[32];
        cudaMemcpy(h_output, d_output, 32 * sizeof(T), cudaMemcpyDeviceToHost);

        printf("Test 1 (basic barrier): ");
        bool pass = true;
        for (int i = 0; i < 32; i++) {
            if (h_output[i] != 31 - i) {
                printf("FAIL at index %d: expected %d, got %d\n", i, 31 - i, (int)h_output[i]);
                pass = false;
                break;
            }
        }
        if (pass) printf("PASS\n");
        all_pass &= pass;

        // Clear output for next test
        cudaMemset(d_output, 0, 32 * sizeof(T));
    }

    // Test 2: Multi-block barrier
    {
        int num_blocks = 4;
        cudaMalloc(&d_output, num_blocks * 32 * sizeof(T));

        test_multi_block_barrier<T><<<num_blocks, 32>>>(d_output);

        T h_output[128];
        cudaMemcpy(h_output, d_output, num_blocks * 32 * sizeof(T), cudaMemcpyDeviceToHost);

        printf("Test 2 (multi-block): ");
        bool pass = true;
        for (int b = 0; b < num_blocks; b++) {
            // Sum of (b*100 + i) for i=0..31 = b*100*32 + 31*32/2 = b*3200 + 496
            T expected_sum = b * 3200 + 496;
            T actual_sum = h_output[b * 32]; // Each block writes sum to first thread's output
            if (actual_sum != expected_sum) {
                printf("FAIL at block %d: expected %d, got %d\n", b, (int)expected_sum, (int)actual_sum);
                pass = false;
                break;
            }
        }
        if (pass) printf("PASS\n");
        all_pass &= pass;

        cudaFree(d_output);
    }

    // Test 3: Nested sync
    {
        cudaMalloc(&d_output, 16 * sizeof(T));
        cudaMemset(d_output, 0, 16 * sizeof(T));  // Clear buffer
        test_nested_sync<T><<<1, 16>>>(d_output);

        T h_output[16];
        cudaMemcpy(h_output, d_output, 16 * sizeof(T), cudaMemcpyDeviceToHost);

        printf("Test 3 (nested sync): ");
        bool pass = true;
        for (int i = 0; i < 16; i++) {
            T expected = i + (i + 1) % 16;
            if (h_output[i] != expected) {
                printf("FAIL at index %d: expected %d, got %d\n", i, (int)expected, (int)h_output[i]);
                pass = false;
                break;
            }
        }
        if (pass) printf("PASS\n");
        all_pass &= pass;

        cudaFree(d_output);
    }

    return all_pass;
}

int main() {
    printf("=== __syncthreads() Barrier Test ===\n");
    bool pass = run_tests<int>();
    printf("=== Overall: %s ===\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
