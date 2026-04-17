#include <cstdio>
#include <cstdlib>

// Test 3: syncthreads with conditional writes
template<typename T>
__global__ void test_sync_conditional(T *data_a, T *data_b, T *output) {
    int tid = threadIdx.x;
    int size = blockDim.x;

    // Initialize output array
    output[tid] = 0;
    __syncthreads();

    // Thread 0 writes to all of data_b
    if (tid == 0) {
        for (int i = 0; i < size; i++) {
            data_b[i] = data_a[i] * 3;
        }
    }
    __syncthreads();

    // All threads read from data_b
    output[tid] = data_b[tid];
}

template<typename T>
bool run_tests() {
    bool all_pass = true;
    const int size = 16;

    // Test 3: syncthreads with conditional writes
    {
        printf("Test 3 (sync with conditional writes): ");
        
        T *h_data_a = (T*)malloc(size * sizeof(T));
        T *h_data_b = (T*)malloc(size * sizeof(T));
        T *h_output = (T*)malloc(size * sizeof(T));
        T *d_data_a, *d_data_b, *d_output;

        // Initialize input data
        for (int i = 0; i < size; i++) {
            h_data_a[i] = i + 1;
            h_data_b[i] = 0;
        }

        cudaMalloc(&d_data_a, size * sizeof(T));
        cudaMalloc(&d_data_b, size * sizeof(T));
        cudaMalloc(&d_output, size * sizeof(T));

        // Initialize device memory via cudaMemcpy
        cudaMemcpy(d_data_a, h_data_a, size * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_data_b, h_data_b, size * sizeof(T), cudaMemcpyHostToDevice);

        test_sync_conditional<T><<<1, size>>>(d_data_a, d_data_b, d_output);

        cudaMemcpy(h_output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost);

        bool pass = true;
        for (int i = 0; i < size; i++) {
            T expected = h_data_a[i] * 3;  // data_b[i] = data_a[i] * 3, then output = data_b[i]
            if (h_output[i] != expected) {
                printf("FAIL at index %d: expected %d, got %d\n", i, (int)expected, (int)h_output[i]);
                pass = false;
                break;
            }
        }
        if (pass) printf("PASS\n");
        all_pass &= pass;

        cudaFree(d_data_a);
        cudaFree(d_data_b);
        cudaFree(d_output);
        free(h_data_a);
        free(h_data_b);
        free(h_output);
    }

    return all_pass;
}

int main() {
    printf("=== Nested __syncthreads() Barrier Test (Global Memory) ===\n");
    bool pass = run_tests<int>();
    printf("=== Overall: %s ===\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
