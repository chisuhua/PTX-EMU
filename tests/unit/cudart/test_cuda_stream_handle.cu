// test_cuda_stream_handle.cpp
// =============================================================================
// Unit test: 验证 cudaStream/cudaEvent handle 正确释放
//
// 审计 §2.2.1 声称 cudaStreamDestroy/cudaEventDestroy 是 STUB，但
// src/cudart/cudart_sim.cpp:692/721 实际有 delete reinterpret_cast<int *>(...)。
// 本测试验证实现正确性。
//
// T1-2 Status: 已隐式完成（审计 §2.2.1 错误）。本测试作为 verification。
// 真实问题（Errata §1.7）:
//   (a) reinterpret_cast<int*> type-unsafe（cudaStream_t 应为 void*）
//   (b) cudaStreamSynchronize:698-703 是 no-op（fake synchronization）
//   (c) cudaEventElapsedTime:741-747 返回硬编码 1.0f
// =============================================================================

#include "catch_amalgamated.hpp"
#include <cuda_runtime.h>

TEST_CASE("cudaStreamCreate/Destroy does not leak handle", "[cudart][stream]") {
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    REQUIRE(stream != nullptr);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("Multiple stream create/destroy cycles", "[cudart][stream]") {
    for (int i = 0; i < 100; ++i) {
        cudaStream_t s;
        REQUIRE(cudaStreamCreate(&s) == cudaSuccess);
        REQUIRE(cudaStreamDestroy(s) == cudaSuccess);
    }
}

TEST_CASE("cudaEventCreate/Destroy does not leak handle", "[cudart][event]") {
    cudaEvent_t event;
    REQUIRE(cudaEventCreate(&event) == cudaSuccess);
    REQUIRE(event != nullptr);
    REQUIRE(cudaEventDestroy(event) == cudaSuccess);
}