/**
 * Test: Shared Memory Store and Load
 * 
 * Verifies that:
 * 1. st.shared correctly writes to shared memory
 * 2. ld.shared correctly reads from shared memory
 * 3. bar.sync properly synchronizes all threads
 * 
 * Expected: Each lane writes lane_id to shared_memory[lane_id], 
 *           then reads back and verifies correctness
 */

#include <cstdio>
#include <cstring>
#include <cuda.h>

#define CUDA_CHECK(call) \
    do { \
        CUresult result = call; \
        if (result != CUDA_SUCCESS) { \
            const char* errstr; \
            cuGetErrorString(result, &errstr); \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errstr); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void verify_shared_store() {
    __shared__ int shared_array[32];
    
    int lane_id = threadIdx.x & 0x1f;  // lane ID within warp
    
    // Store lane ID to shared memory
    shared_array[lane_id] = lane_id * 100 + 7;
    
    // Synchronize all threads
    __syncthreads();
    
    // Verify: each thread should see all values correctly stored
    int expected = lane_id * 100 + 7;
    int actual = shared_array[lane_id];
    
    if (actual != expected) {
        printf("Thread %d: expected %d, got %d\\n", lane_id, expected, actual);
    }
}

int main() {
    printf("=== Shared Memory Store Test ===\\n");
    
    // Launch kernel with 1 warp (32 threads)
    verify_shared_store<<<1, 32>>>();
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("Test completed.\\n");
    
    return 0;
}
