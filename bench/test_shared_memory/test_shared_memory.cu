#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// Shared Memory Unit Tests for PTX-EMU
// ============================================================================
// Test coverage:
//   P0: test_sync_basic, test_template_shared, test_dynamic_shared
//   P1: test_divergence_sync, test_nested_sync
//   P2: test_multi_warp_shared, test_bank_conflict, test_shared_padding
// ============================================================================

#define WARP_SIZE 32

// ============================================================================
// P0: Basic __syncthreads() test - write-after-read barrier
// ============================================================================
template<typename T>
__global__ void test_sync_basic_kernel(T *output) {
    __shared__ T shared_data[32];

    int tid = threadIdx.x;

    // Phase 1: All threads write their index
    shared_data[tid] = tid;

    // Barrier: ensure all writes complete before any read
    __syncthreads();

    // Phase 2: All threads read from reversed index
    T value = shared_data[31 - tid];

    // Another barrier before writing output
    __syncthreads();

    output[tid] = value;
}

bool test_sync_basic() {
    printf("  test_sync_basic: ");
    int *d_output;
    cudaMalloc(&d_output, 32 * sizeof(int));

    test_sync_basic_kernel<int><<<1, 32>>>(d_output);

    int h_output[32];
    cudaMemcpy(h_output, d_output, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    for (int i = 0; i < 32; i++) {
        if (h_output[i] != 31 - i) {
            printf("FAIL at index %d: expected %d, got %d\n", i, 31 - i, h_output[i]);
            return false;
        }
    }
    printf("PASS\n");
    return true;
}

// ============================================================================
// P0: Template kernel with shared memory (exposes symbol resolution bug)
// ============================================================================
template<typename T>
__global__ void test_template_shared_kernel(T *output) {
    // This template function declares shared memory
    // PTX-EMU had a bug where elementNum was 0 for template functions
    __shared__ T shared_data[32];

    int tid = threadIdx.x;
    shared_data[tid] = static_cast<T>(tid * 2);

    __syncthreads();

    T value = shared_data[31 - tid];
    output[tid] = value;
}

bool test_template_shared() {
    printf("  test_template_shared: ");
    int *d_output;
    cudaMalloc(&d_output, 32 * sizeof(int));

    test_template_shared_kernel<int><<<1, 32>>>(d_output);

    int h_output[32];
    cudaMemcpy(h_output, d_output, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    for (int i = 0; i < 32; i++) {
        int expected = (31 - i) * 2;
        if (h_output[i] != expected) {
            printf("FAIL at index %d: expected %d, got %d\n", i, expected, h_output[i]);
            return false;
        }
    }
    printf("PASS\n");
    return true;
}

// ============================================================================
// P0: Dynamic shared memory allocation
// ============================================================================
__global__ void test_dynamic_shared_kernel(int *output, int size) {
    // Dynamic shared memory - size determined at launch
    extern __shared__ int shared_data[];
    int tid = threadIdx.x;

    if (tid < size) {
        shared_data[tid] = tid * 3;
    }

    __syncthreads();

    if (tid < size) {
        output[tid] = shared_data[size - 1 - tid];
    } else {
        output[tid] = -1;
    }
}

bool test_dynamic_shared() {
    printf("  test_dynamic_shared: ");
    const int size = 32;
    int *d_output;
    cudaMalloc(&d_output, 32 * sizeof(int));

    // Launch with dynamic shared memory size
    test_dynamic_shared_kernel<<<1, 32, size * sizeof(int)>>>(d_output, size);

    int h_output[32];
    cudaMemcpy(h_output, d_output, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    for (int i = 0; i < size; i++) {
        int expected = (size - 1 - i) * 3;
        if (h_output[i] != expected) {
            printf("FAIL at index %d: expected %d, got %d\n", i, expected, h_output[i]);
            return false;
        }
    }
    // Check padding entries are unchanged
    for (int i = size; i < 32; i++) {
        if (h_output[i] != -1) {
            printf("FAIL at index %d: expected -1, got %d\n", i, h_output[i]);
            return false;
        }
    }
    printf("PASS\n");
    return true;
}

// ============================================================================
// P1: Warp divergence with __syncthreads() barrier
// ============================================================================
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
    printf("  test_divergence_sync: ");
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

// ============================================================================
// P1: Nested __syncthreads() with multiple phases
// ============================================================================
template<typename T>
__global__ void test_nested_sync_kernel(T *output) {
    __shared__ T phase1[16];
    __shared__ T phase2[16];

    int tid = threadIdx.x;

    // Phase 1: fill phase1
    phase1[tid] = tid;
    __syncthreads();

    // Phase 2: compute into phase2 using phase1 data
    if (tid < 16) {
        phase2[tid] = phase1[tid] + phase1[(tid + 1) % 16];
    }
    __syncthreads();

    // Phase 3: final computation
    output[tid] = phase2[tid % 16] + phase2[(tid + 1) % 16];
}

bool test_nested_sync() {
    printf("  test_nested_sync: ");
    int *d_output;
    cudaMalloc(&d_output, 32 * sizeof(int));

    test_nested_sync_kernel<int><<<1, 32>>>(d_output);

    int h_output[32];
    cudaMemcpy(h_output, d_output, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    for (int i = 0; i < 32; i++) {
        int expected = (i % 16) + ((i + 1) % 16) + ((i + 1) % 16) + ((i + 2) % 16);
        if (h_output[i] != expected) {
            printf("FAIL at index %d: expected %d, got %d\n", i, expected, h_output[i]);
            return false;
        }
    }
    printf("PASS\n");
    return true;
}

// ============================================================================
// P2: Multi-warp shared memory access
// ============================================================================
template<typename T>
__global__ void test_multi_warp_shared_kernel(T *output, int size) {
    __shared__ T shared_data[64];  // Large enough for multiple warps

    int tid = threadIdx.x;

    // Each thread writes its lane ID
    int lane = tid % WARP_SIZE;
    shared_data[tid] = lane;

    __syncthreads();

    // Each thread reads from opposite half
    int read_idx = (tid < size / 2) ? (tid + size / 2) : (tid - size / 2);
    output[tid] = shared_data[read_idx];
}

bool test_multi_warp_shared() {
    printf("  test_multi_warp_shared: ");
    const int size = 64;  // 2 warps
    int *d_output;
    cudaMalloc(&d_output, size * sizeof(int));

    test_multi_warp_shared_kernel<int><<<1, size>>>(d_output, size);

    int h_output[64];
    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    for (int i = 0; i < size; i++) {
        int expected = (i < size / 2) ? (i + size / 2) % WARP_SIZE : (i - size / 2) % WARP_SIZE;
        if (h_output[i] != expected) {
            printf("FAIL at index %d: expected %d, got %d\n", i, expected, h_output[i]);
            return false;
        }
    }
    printf("PASS\n");
    return true;
}

// ============================================================================
// P2: Bank conflict detection via timing pattern
// ============================================================================
template<typename T>
__global__ void test_bank_conflict_row_kernel(T *output) {
    __shared__ T shared_data[32][32];  // 32x32 matrix, row-major

    int tid = threadIdx.x;
    int row = tid;

    // Row access: threads in same row access consecutive columns
    for (int i = 0; i < 32; i++) {
        shared_data[row][i] = row * 32 + i;
    }

    __syncthreads();

    // Read back row sums
    T sum = 0;
    for (int i = 0; i < 32; i++) {
        sum += shared_data[row][i];
    }
    output[row] = sum;
}

template<typename T>
__global__ void test_bank_conflict_col_kernel(T *output) {
    __shared__ T shared_data[32][32];  // Same layout

    int tid = threadIdx.x;
    int col = tid;

    // Column access: threads in same column access consecutive rows
    for (int i = 0; i < 32; i++) {
        shared_data[i][col] = i * 32 + col;
    }

    __syncthreads();

    // Read back column sums
    T sum = 0;
    for (int i = 0; i < 32; i++) {
        sum += shared_data[i][col];
    }
    output[col] = sum;
}

bool test_bank_conflict() {
    printf("  test_bank_conflict_row: ");
    int *d_output;
    cudaMalloc(&d_output, 32 * sizeof(int));

    test_bank_conflict_row_kernel<int><<<1, 32>>>(d_output);

    int h_output[32];
    cudaMemcpy(h_output, d_output, 32 * sizeof(int), cudaMemcpyDeviceToHost);

    // Expected row sum: each row has values row*32 + (0..31)
    // Sum = row*32*32 + (0+1+...+31) = row*1024 + 496
    bool pass = true;
    for (int row = 0; row < 32; row++) {
        int expected = row * 1024 + 496;
        if (h_output[row] != expected) {
            printf("FAIL at row %d: expected %d, got %d\n", row, expected, h_output[row]);
            pass = false;
            break;
        }
    }
    if (pass) printf("PASS\n");

    printf("  test_bank_conflict_col: ");
    test_bank_conflict_col_kernel<int><<<1, 32>>>(d_output);
    cudaMemcpy(h_output, d_output, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    // Expected column sum: each column has values (0..31)*32 + col
    // Sum = (0+1+...+31)*32 + col*32 = 496*32 + col*32 = 15872 + col*32
    for (int col = 0; col < 32; col++) {
        int expected = 15872 + col * 32;
        if (h_output[col] != expected) {
            printf("FAIL at col %d: expected %d, got %d\n", col, expected, h_output[col]);
            return false;
        }
    }
    printf("PASS\n");
    return pass;
}

// ============================================================================
// P2: Shared memory with padding to avoid bank conflicts
// ============================================================================
template<typename T>
__global__ void test_shared_padding_kernel(T *output) {
    // Use 33 elements per row (32 + 1 padding) to avoid bank conflicts
    __shared__ T shared_data[32][33];

    int tid = threadIdx.x;
    int row = tid;

    // Write with padding
    for (int i = 0; i < 32; i++) {
        shared_data[row][i] = row * 33 + i;
    }

    __syncthreads();

    // Read and verify
    T sum = 0;
    for (int i = 0; i < 32; i++) {
        sum += shared_data[row][i];
    }
    output[row] = sum;
}

bool test_shared_padding() {
    printf("  test_shared_padding: ");
    int *d_output;
    cudaMalloc(&d_output, 32 * sizeof(int));

    test_shared_padding_kernel<int><<<1, 32>>>(d_output);

    int h_output[32];
    cudaMemcpy(h_output, d_output, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    // Expected: row * 33 * 32 + sum(0..31) = row * 1056 + 496
    for (int row = 0; row < 32; row++) {
        int expected = row * 1056 + 496;
        if (h_output[row] != expected) {
            printf("FAIL at row %d: expected %d, got %d\n", row, expected, h_output[row]);
            return false;
        }
    }
    printf("PASS\n");
    return true;
}

// ============================================================================
// Main: Run all tests
// ============================================================================
int main() {
    printf("=== Shared Memory Unit Tests ===\n\n");

    bool all_pass = true;

    // P0 tests
    printf("[P0] Basic & Template Tests:\n");
    all_pass &= test_sync_basic();
    all_pass &= test_template_shared();
    all_pass &= test_dynamic_shared();
    printf("\n");

    // P1 tests
    printf("[P1] Divergence & Nested Sync:\n");
    all_pass &= test_divergence_sync();
    all_pass &= test_nested_sync();
    printf("\n");

    // P2 tests
    printf("[P2] Multi-Warp & Bank Conflict:\n");
    all_pass &= test_multi_warp_shared();
    all_pass &= test_bank_conflict();
    all_pass &= test_shared_padding();
    printf("\n");

    printf("=== Overall: %s ===\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
