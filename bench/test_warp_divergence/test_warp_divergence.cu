#include <cstdio>
#include <cstdlib>

// Test 1: Simple branch divergence within a warp
// Threads 0-15 take path A, threads 16-31 take path B
template<typename T>
__global__ void test_simple_divergence(T *output) {
    int tid = threadIdx.x;
    T value;

    if (tid < 16) {
        // Path A: compute square
        value = tid * tid;
    } else {
        // Path B: compute cube
        value = tid * tid * tid;
    }

    output[tid] = value;
}

// Test 2: Divergence with synchronization after branches
// Each warp processes a portion of data, branches differently, then syncs
template<typename T>
__global__ void test_divergence_with_sync(T *output, int size) {
    __shared__ T shared_data[32];

    int tid = threadIdx.x;
    int lane = tid % 32;

    T value;
    if (lane < 16) {
        // First half of warp: accumulate
        value = 0;
        for (int i = 0; i < lane + 1; i++) {
            value += i;
        }
    } else {
        // Second half of warp: multiply
        value = 1;
        for (int i = 1; i <= lane - 15; i++) {
            value *= i;
        }
    }

    shared_data[lane] = value;
    __syncthreads();

    // Thread 0 collects the result
    if (tid == 0) {
        T sum = 0;
        for (int i = 0; i < 32; i++) {
            sum += shared_data[i];
        }
        output[0] = sum;
    } else {
        output[tid] = 0;
    }
}

// Test 3: Nested divergence (multiple branch levels)
template<typename T>
__global__ void test_nested_divergence(T *output) {
    int tid = threadIdx.x;

    T value = tid;

    // First level: branch on tid % 2
    if (tid % 2 == 0) {
        value += 100;  // Even path
    } else {
        value += 200;  // Odd path
    }

    // Second level (within each branch): branch on tid % 4
    if (tid % 4 < 2) {
        value += 10;   // Inner path A
    } else {
        value += 20;   // Inner path B
    }

    output[tid] = value;
}

// Test 4: Divergence with reduction pattern
template<typename T>
__global__ void test_divergence_reduction(T *output) {
    __shared__ T warp_results[32];

    int tid = threadIdx.x;
    int lane = tid % 32;

    T value = tid + 1;  // 1 to 32

    // Each lane computes different reduction step based on lane ID
    if (lane < 16) {
        value += warp_results[lane + 16];
    }

    warp_results[lane] = value;
    __syncthreads();

    if (tid == 0) {
        output[0] = warp_results[0];
    } else {
        output[tid] = 0;
    }
}

template<typename T>
bool run_tests() {
    bool all_pass = true;

    // Test 1: Simple divergence
    {
        T *d_output;
        cudaMalloc(&d_output, 32 * sizeof(T));
        test_simple_divergence<T><<<1, 32>>>(d_output);

        T h_output[32];
        cudaMemcpy(h_output, d_output, 32 * sizeof(T), cudaMemcpyDeviceToHost);

        printf("Test 1 (simple divergence): ");
        bool pass = true;
        for (int i = 0; i < 16; i++) {
            T expected = i * i;
            if (h_output[i] != expected) {
                printf("FAIL at lane %d (path A): expected %d, got %d\n", i, (int)expected, (int)h_output[i]);
                pass = false;
                break;
            }
        }
        for (int i = 16; i < 32; i++) {
            T expected = i * i * i;
            if (h_output[i] != expected) {
                printf("FAIL at lane %d (path B): expected %d, got %d\n", i, (int)expected, (int)h_output[i]);
                pass = false;
                break;
            }
        }
        if (pass) printf("PASS\n");
        all_pass &= pass;

        cudaFree(d_output);
    }

    // Test 2: Divergence with synchronization
    {
        T *d_output;
        cudaMalloc(&d_output, 32 * sizeof(T));
        test_divergence_with_sync<T><<<1, 32>>>(d_output, 32);

        T h_output[32];
        cudaMemcpy(h_output, d_output, 32 * sizeof(T), cudaMemcpyDeviceToHost);

        printf("Test 2 (divergence + sync): ");
        bool pass = true;
        // Check first thread has sum of all shared_data values
        T expected_sum = 0;
        for (int i = 0; i < 16; i++) {
            expected_sum += (i * (i + 1)) / 2;  // Sum of 0..i
        }
        for (int i = 16; i < 32; i++) {
            // Factorial-like product for lane > 16
            T prod = 1;
            for (int j = 1; j <= i - 15; j++) {
                prod *= j;
            }
            expected_sum += prod;
        }
        if (h_output[0] != expected_sum) {
            printf("FAIL: expected %d, got %d\n", (int)expected_sum, (int)h_output[0]);
            pass = false;
        }
        if (pass) printf("PASS\n");
        all_pass &= pass;

        cudaFree(d_output);
    }

    // Test 3: Nested divergence
    {
        T *d_output;
        cudaMalloc(&d_output, 32 * sizeof(T));
        test_nested_divergence<T><<<1, 32>>>(d_output);

        T h_output[32];
        cudaMemcpy(h_output, d_output, 32 * sizeof(T), cudaMemcpyDeviceToHost);

        printf("Test 3 (nested divergence): ");
        bool pass = true;
        for (int i = 0; i < 32; i++) {
            T expected = i;
            expected += (i % 2 == 0) ? 100 : 200;  // First branch
            expected += (i % 4 < 2) ? 10 : 20;       // Second branch
            if (h_output[i] != expected) {
                printf("FAIL at lane %d: expected %d, got %d\n", i, (int)expected, (int)h_output[i]);
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
    printf("=== Warp Divergence Test ===\n");
    bool pass = run_tests<int>();
    printf("=== Overall: %s ===\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
