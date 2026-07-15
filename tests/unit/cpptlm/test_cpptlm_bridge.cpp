// test_cpptlm_bridge.cpp
// =============================================================================
// Unit test: CppTLM Bridge ABI 接口验证 (D-PTX-1 + Task #1)
//
// 验证 cpptlm_bridge.h 的 5 个纯虚方法签名 + 全局指针 + static_assert
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/cpptlm_bridge.h"
#include <cstring>

// Mock Bridge 实现用于测试
class MockCppTLMBridge : public CppTLMBridge {
public:
    int version() const override { return CPPTLMBRIDGE_VERSION; }

    int submit_kernel(uint64_t kernel_id, const char* kernel_name,
                     uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                     uint32_t block_x, uint32_t block_y, uint32_t block_z,
                     const void** kernel_args, size_t args_count,
                     size_t shared_mem, uint64_t stream_id) override {
        last_submit_id = kernel_id;
        last_submit_name = kernel_name;
        return 0;  // success
    }

    uint64_t poll_kernel(uint64_t kernel_id) override {
        return 0;  // completed
    }

    int synchronize_stream(uint64_t stream_id) override {
        return 0;  // success
    }

    uint64_t global_access(uint64_t device_addr, uint64_t val, uint8_t type) override {
        return 100;  // 100 cycles latency
    }

    // Test inspection
    uint64_t last_submit_id = 0;
    const char* last_submit_name = nullptr;
};

TEST_CASE("CppTLMBridge version() returns CPPTLMBRIDGE_VERSION", "[cpptlm][bridge]") {
    MockCppTLMBridge bridge;
    REQUIRE(bridge.version() == CPPTLMBRIDGE_VERSION);
    REQUIRE(CPPTLMBRIDGE_VERSION == 1);
}

TEST_CASE("CppTLMBridge submit_kernel() accepts 12 parameters", "[cpptlm][bridge]") {
    MockCppTLMBridge bridge;

    uint64_t kernel_id = 42;
    const char* kernel_name = "test_kernel";
    uint32_t grid[3] = {2, 2, 1};
    uint32_t block[3] = {32, 1, 1};
    const void* args[2] = {nullptr, nullptr};
    size_t args_count = 2;
    size_t shared_mem = 1024;
    uint64_t stream_id = 0;

    int result = bridge.submit_kernel(kernel_id, kernel_name,
                                      grid[0], grid[1], grid[2],
                                      block[0], block[1], block[2],
                                      args, args_count,
                                      shared_mem, stream_id);

    REQUIRE(result == 0);
    REQUIRE(bridge.last_submit_id == kernel_id);
    REQUIRE(std::string(bridge.last_submit_name) == kernel_name);
}

TEST_CASE("CppTLMBridge poll_kernel() returns completion status", "[cpptlm][bridge]") {
    MockCppTLMBridge bridge;

    // 0 = completed
    REQUIRE(bridge.poll_kernel(1) == 0);

    // >0 = remaining cycles (mock always returns 0)
    // UINT64_MAX = unknown kernel (mock doesn't implement this)
}

TEST_CASE("CppTLMBridge synchronize_stream() succeeds", "[cpptlm][bridge]") {
    MockCppTLMBridge bridge;

    int result = bridge.synchronize_stream(0);
    REQUIRE(result == 0);
}

TEST_CASE("CppTLMBridge global_access() returns latency", "[cpptlm][bridge]") {
    MockCppTLMBridge bridge;

    uint64_t device_addr = 0x1000;
    uint64_t val = 0;
    uint8_t type = 0;  // LD

    uint64_t latency = bridge.global_access(device_addr, val, type);
    REQUIRE(latency == 100);  // mock returns 100 cycles
}

TEST_CASE("g_cpptlm_bridge global pointer defaults to nullptr", "[cpptlm][bridge]") {
    // g_cpptlm_bridge is declared in cudart_sim.cpp
    // When not loaded, it should be nullptr
    extern CppTLMBridge* g_cpptlm_bridge;
    REQUIRE(g_cpptlm_bridge == nullptr);
}

TEST_CASE("static_assert: cudaStream_t fits in uint64_t", "[cpptlm][bridge]") {
    // This test verifies the static_assert in cpptlm_bridge.h compiles
    // If cudaStream_t > uint64_t, compilation would fail
    REQUIRE(sizeof(cudaStream_t) <= sizeof(uint64_t));
}
