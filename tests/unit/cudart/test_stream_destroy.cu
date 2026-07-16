// test_stream_destroy.cpp
// =============================================================================
// Unit test: cudaStreamDestroy correctness (B3)
//
// Per Metis second-pass review (B3): cudaStreamDestroy was calling
// `delete reinterpret_cast<int *>(stream)` on a uint64_t decoded stream
// handle — undefined behavior (SEGFAULT in unit_cuda_stream_handle).
// Also failed to clean up g_active_streams, causing the set to grow unboundedly.
//
// This test verifies:
//   1. cudaStreamDestroy(stream) does not crash (no UB delete)
//   2. cudaStreamDestroy(nullptr) is a safe no-op (default stream)
//   3. Repeated create/destroy cycles do not crash
//
// Note: g_active_streams cleanup is verified indirectly — the SEGFAULT
// in the existing unit_cuda_stream_handle test is the primary regression
// guard. This test adds explicit destroy-safety coverage.
//
// Ref: docs/adr/0021-cpptlm-d1-full-integration.md
//      docs/dev-process/lessons-learned.md §2 (state-modification-audit)
// =============================================================================

#include "catch_amalgamated.hpp"
#include <cuda_runtime.h>

TEST_CASE("cudaStreamDestroy on created stream does not crash", "[cudart][stream][destroy]") {
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    REQUIRE(stream != nullptr);

    // Previously this called `delete reinterpret_cast<int *>(stream)` — UB.
    // Must return cudaSuccess without crashing.
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("cudaStreamDestroy on default stream (nullptr) is a no-op", "[cudart][stream][destroy]") {
    // Per CUDA spec: destroying the default stream (nullptr) is a no-op.
    REQUIRE(cudaStreamDestroy(nullptr) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(0)) == cudaSuccess);
}

TEST_CASE("Repeated stream create/destroy cycles do not crash", "[cudart][stream][destroy]") {
    // Stress test: 200 cycles. Previously each cycle called `delete` on a
    // non-heap pointer — would SEGFAULT reliably within a few iterations
    // under AddressSanitizer or even uninstrumented builds.
    for (int i = 0; i < 200; ++i) {
        cudaStream_t s;
        REQUIRE(cudaStreamCreate(&s) == cudaSuccess);
        REQUIRE(s != nullptr);
        REQUIRE(cudaStreamDestroy(s) == cudaSuccess);
    }
}

TEST_CASE("Double-destroy of a stream handle is safe", "[cudart][stream][destroy]") {
    // CUDA spec does not guarantee double-destroy safety, but PTX-EMU's
    // fake runtime should not crash (it tracks via g_active_streams and
    // erase is idempotent on std::unordered_set). We verify no crash;
    // the return code is not asserted (CUDA would return errorInvalidHandle).
    cudaStream_t s;
    REQUIRE(cudaStreamCreate(&s) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(s) == cudaSuccess);
    // Second destroy — must not crash. Return code unspecified (we accept any).
    (void)cudaStreamDestroy(s);
}
