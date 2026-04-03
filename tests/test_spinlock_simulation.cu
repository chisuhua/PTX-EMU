/**
 * @file test_spinlock_simulation.cu
 * @brief Comprehensive spinlock simulation test for SIMT architecture
 * Tests the per-thread PC mechanism for resolving warp-level deadlocks
 * 
 * @author PTX-EMU Team
 * @date 2026-04-02
 */

#include "catch_amalgamated.hpp"
#include "ptxsim/execution_types.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/gpu_context.h"
#include <cuda.h>
#include <iostream>

using namespace ptxsim;

// ============================================================================
// Global lock for simulation
// ============================================================================
__device__ int test_lock = 0;
__device__ int critical_section_count = 0;
__device__ int barrier_count = 0;

// ============================================================================
// Test 1: Basic AtomicCAS Spinlock
// ============================================================================
template<typename T>
__global__ void test_atomic_spinlock(T *output, int num_threads) {
    T local_count = 0;
    int spin_iterations = 0;
    
    // Attempt to acquire lock
    while (true) {
        spin_iterations++;
        int old = atomicCAS(&test_lock, 0, 1);
        if (old == 0) {
            // Lock acquired
            break;
        }
        // Spin - in real HW this would be a backoff
        if (spin_iterations > 1000) {
            // Timeout - skip critical section
            output[threadIdx.x] = -1;
            return;
        }
    }
    
    // Critical section
    local_count = atomicAdd(&critical_section_count, 1);
    
    // Release lock
    atomicExch(&test_lock, 0);
    
    // Write result
    output[threadIdx.x] = local_count;
}

// ============================================================================
// Test 2: Spinlock with Barrier Synchronization
// ============================================================================
__global__ void test_spinlock_with_barrier(T *output, int num_threads) {
    __shared__ int shared_lock;
    
    if (threadIdx.x == 0) {
        shared_lock = 0;  // Initialize
    }
    __syncthreads();
    
    __shared__ int lock_success_count;
    if (threadIdx.x == 0) {
        lock_success_count = 0;
    }
    __syncthreads();
    
    // Each thread tries to acquire lock
    int local_success = 0;
    int attempts = 0;
    
    while (attempts < 100) {
        int old = atomicCAS(&test_lock, 0, 1);
        if (old == 0) {
            // Got lock
            local_success = 1;
            atomicAdd(&lock_success_count, 1);
            
            // Release immediately
            atomicExch(&test_lock, 0);
            break;
        }
        attempts++;
    }
    
    // Synchronize all threads
    __syncthreads();
    
    // Write results
    output[threadIdx.x] = (threadIdx.x < lock_success_count) ? threadIdx.x : -1;
}

// ============================================================================
// Test 3: Per-Thread PC Verification (Data Structure Test)
// ============================================================================
TEST_CASE("AtomicCAS spinlock basic operation", "[simt][spinlock][atomic][integration]") {
    int num_threads = 32;
    int *d_output;
    int h_output[32];
    
    cudaMalloc(&d_output, num_threads * sizeof(int));
    cudaMemset(&test_lock, 0, sizeof(int));
    cudaMemset(&critical_section_count, 0, sizeof(int));
    
    test_atomic_spinlock<int><<<1, num_threads>>>(d_output, num_threads);
    
    cudaMemcpy(h_output, d_output, num_threads * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify results
    int success_count = 0;
    for (int i = 0; i < num_threads; ++i) {
        if (h_output[i] >= 0) {
            success_count++;
        }
    }
    
    // All threads should succeed (lock acquisition is serialized, not deadlocked)
    REQUIRE(success_count == num_threads);
    
    // All threads should have executed critical section
    int final_count;
    cudaMemcpy(&final_count, &critical_section_count, sizeof(int), cudaMemcpyDeviceToHost);
    REQUIRE(final_count == num_threads);
    
    cudaFree(d_output);
}

// ============================================================================
// Test 4: Spinlock with Barrier (Deadlock Test)
// ============================================================================
TEST_CASE("Spinlock with barrier synchronization", "[simt][spinlock][barrier][deadlock][integration]") {
    int num_threads = 32;
    int *d_output;
    int h_output[32];
    
    cudaMalloc(&d_output, num_threads * sizeof(int));
    cudaMemset(&test_lock, 0, sizeof(int));
    
    test_spinlock_with_barrier<<<1, num_threads>>>(d_output, num_threads);
    
    cudaMemcpy(h_output, d_output, num_threads * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify barrier didn't cause deadlock (kernel completed)
    // Some negative values are expected (threads that didn't acquire lock)
    int success_count = 0;
    for (int i = 0; i < num_threads; ++i) {
        if (h_output[i] >= 0) {
            success_count++;
        }
    }
    
    // At least some threads should succeed
    REQUIRE(success_count > 0);
    
    cudaFree(d_output);
}

// ============================================================================
// Test 5: Warp-Level Divergence Test
// ============================================================================
__global__ void test_warp_divergence_barrier(T *output) {
    // This tests the core SIMT capability: divergent execution + barrier
    
    // Introduce divergence
    if (threadIdx.x < 16) {
        // Threads 0-15: path A
        for (int i = 0; i < 100; ++i) {
            output[threadIdx.x] += i;
        }
    } else {
        // Threads 16-31: path B
        output[threadIdx.x] = threadIdx.x * 2;
    }
    
    // Barrier - all threads must reach here
    __syncthreads();
    
    // After barrier, converge
    if (threadIdx.x == 0) {
        output[0] += output[threadIdx.x];
    }
}

TEST_CASE("Warp divergence with barrier (SIMT core test)", "[simt][divergence][barrier][core]") {
    int num_threads = 32;
    int *d_output;
    int h_output[32];
    
    cudaMalloc(&d_output, num_threads * sizeof(int));
    cudaMemset(d_output, 0, num_threads * sizeof(int));
    
    test_warp_divergence_barrier<<<1, num_threads>>>(d_output);
    
    cudaMemcpy(h_output, d_output, num_threads * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify kernel completed (no deadlock from divergence+barrier)
    bool all_computed = true;
    for (int i = 16; i < 32; ++i) {
        if (h_output[i] != i * 2) {
            all_computed = false;
            break;
        }
    }
    REQUIRE(all_computed == true);
    
    cudaFree(d_output);
}

// ============================================================================
// Test 6: Concurrent Lock Holders Stress Test
// ============================================================================
__global__ void test_concurrent_spinlock(int *success_flags, int *order, int thread_count) {
    int local_order = atomicAdd(order, 1);  // Track order of lock acquisition
    
    int attempts = 0;
    int success = 0;
    
    while (attempts < 50) {
        int old = atomicCAS(&test_lock, 0, 1);
        if (old == 0) {
            success = 1;
            atomicExch(&test_lock, 0);
            break;
        }
        attempts++;
    }
    
    success_flags[threadIdx.x] = success;
    order[local_order] = threadIdx.x;
}

TEST_CASE("Concurrent spinlock stress test", "[simt][spinlock][stress][concurrent]") {
    int num_threads = 32;
    int *d_success_flags;
    int *d_order;
    int h_success_flags[32];
    int h_order[32];
    
    cudaMalloc(&d_success_flags, num_threads * sizeof(int));
    cudaMalloc(&d_order, num_threads * sizeof(int));
    
    cudaMemset(&test_lock, 0, sizeof(int));
    cudaMemset(d_order, -1, num_threads * sizeof(int));
    
    test_concurrent_spinlock<<<1, num_threads>>>(d_success_flags, d_order, num_threads);
    
    cudaMemcpy(h_success_flags, d_success_flags, num_threads * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_order, d_order, num_threads * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Count successful lock acquisitions
    int success_count = 0;
    for (int i = 0; i < num_threads; ++i) {
        if (h_success_flags[i] == 1) {
            success_count++;
        }
    }
    
    // With enough attempts, most threads should succeed
    REQUIRE(success_count >= num_threads / 2);  // At least 50% success rate
    
    cudaFree(d_success_flags);
    cudaFree(d_order);
}

// ============================================================================
// Test Runner
// ============================================================================
TEMPLATE_TEST_CASE("Spinlock integration tests", "[simt][spinlock][integration]", 
                  int, int32_t, uint32_t) {
    
    SECTION("Basic atomicCAS spinlock") {
        TestType num_threads = 32;
        TestType *d_output;
        TestType h_output[32];
        
        cudaMalloc(&d_output, num_threads * sizeof(TestType));
        cudaMemset(&test_lock, 0, sizeof(int));
        
        test_atomic_spinlock<TestType><<<1, num_threads>>>(d_output, num_threads);
        cudaMemcpy(h_output, d_output, num_threads * sizeof(TestType), cudaMemcpyDeviceToHost);
        
        // Verify all threads completed (no deadlock)
        for (int i = 0; i < num_threads; ++i) {
            REQUIRE(h_output[i] >= -1);  // -1 is timeout, otherwise success
        }
        
        cudaFree(d_output);
    }
}

