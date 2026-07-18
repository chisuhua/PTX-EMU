// test_cudart_stream.cpp
// =============================================================================
// Unit test: CUDA Runtime Stream API supplementary tests
//
// Complements the existing 3 cudart stream tests:
//   - test_cuda_stream_handle.cu (create/destroy lifecycle, event create/destroy)
//   - test_stream_destroy.cu     (destroy safety, nullptr, double-destroy)
//   - test_stream_sync_loop.cpp  (sync polling loop with mock bridge)
//
// This file adds:
//   - Uniqueness: different cudaStreamCreate calls produce different handles
//   - No-leak: destroy then create again works without resource exhaustion
//   - Sync contract: synchronize on a freshly created stream returns cudaSuccess
//   - Null pointer: cudaStreamCreate(nullptr) returns cudaErrorInvalidValue
//
// Uses project-internal cudart_intrinsics.h instead of <cuda_runtime.h>
// because this file is compiled with g++ (not nvcc).
//
// Ref: ADR-0010 (add-cudart-unit-test-coverage)
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/cudart_intrinsics.h"

// CUDA runtime entry points (C linkage, defined in cudart_sim.cpp).
extern "C" {
cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
}

// =============================================================================
// cudaStreamCreate
// =============================================================================

TEST_CASE("cudaStreamCreate returns non-null unique streams",
          "[cudart][stream][create]") {
    cudaStream_t s1 = nullptr;
    cudaStream_t s2 = nullptr;
    REQUIRE(cudaStreamCreate(&s1) == cudaSuccess);
    REQUIRE(cudaStreamCreate(&s2) == cudaSuccess);
    REQUIRE(s1 != nullptr);
    REQUIRE(s2 != nullptr);
    REQUIRE(s1 != s2);  // different calls must produce different handles

    REQUIRE(cudaStreamDestroy(s1) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(s2) == cudaSuccess);
}

TEST_CASE("cudaStreamCreate null pointer returns InvalidValue",
          "[cudart][stream][create]") {
    REQUIRE(cudaStreamCreate(nullptr) == cudaErrorInvalidValue);
}

// =============================================================================
// cudaStreamDestroy + recreate (no leak)
// =============================================================================

TEST_CASE("Destroy then create stream does not leak",
          "[cudart][stream][destroy]") {
    // Create and destroy a stream, then create another.
    // If g_active_streams was not cleaned up, repeated cycles would
    // accumulate stale entries (verified indirectly — no crash or hang).
    cudaStream_t s1 = nullptr;
    REQUIRE(cudaStreamCreate(&s1) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(s1) == cudaSuccess);

    cudaStream_t s2 = nullptr;
    REQUIRE(cudaStreamCreate(&s2) == cudaSuccess);
    REQUIRE(s2 != nullptr);
    REQUIRE(cudaStreamDestroy(s2) == cudaSuccess);
}

// =============================================================================
// cudaStreamSynchronize
// =============================================================================

TEST_CASE("cudaStreamSynchronize on fresh stream returns Success",
          "[cudart][stream][sync]") {
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    // No kernels submitted — sync should return immediately with success.
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("cudaStreamSynchronize on default stream (nullptr) returns Success",
          "[cudart][stream][sync]") {
    // Default stream is nullptr; sync on it should succeed (no-op).
    REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);
}
