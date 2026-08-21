// tests/integration/cudart/test_sync_only_immediate.cpp
// Sync-only runtime contract: cudaStreamSynchronize(nullptr) returns cudaSuccess
// immediately because sync-mode cudaLaunchKernel already completed via
// wait_for_completion() inside the launch call.
// Per cudart-sync-only-runtime/spec.md Stream lifecycle scenario.
// Created per Metis 2026-08-21 MUST-RESOLVE #5 (Phase 2a replacement for
// deleted test_stream_sync_loop.cpp which tested the same contract via
// g_cpptlm_bridge == nullptr path).

#include <catch_amalgamated.hpp>

extern "C" {
    typedef int cudaError_t;
    cudaError_t cudaStreamSynchronize(void* stream);
}

TEST_CASE("cudaStreamSynchronize(nullptr) returns cudaSuccess immediately",
          "[integration][cudart][sync]") {
    cudaError_t err = cudaStreamSynchronize(nullptr);
    REQUIRE(err == 0);
}