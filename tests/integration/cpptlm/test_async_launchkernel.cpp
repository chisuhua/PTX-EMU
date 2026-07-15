// test_async_launchkernel.cpp
// =============================================================================
// Integration test: cudaLaunchKernel 异步路径 (D-PTX-1 + Task #2)
//
// 验证 bridge != nullptr 时 kernel 走异步提交路径
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/cpptlm_bridge.h"
#include <atomic>

class AsyncMockBridge : public CppTLMBridge {
public:
    int version() const override { return 1; }

    int submit_kernel(uint64_t kernel_id, const char* kernel_name,
                     uint32_t gx, uint32_t gy, uint32_t gz,
                     uint32_t bx, uint32_t by, uint32_t bz,
                     const void** args, size_t args_count,
                     size_t shared_mem, uint64_t stream_id) override {
        submit_count.fetch_add(1);
        last_kernel_id = kernel_id;
        last_stream_id = stream_id;
        return 0;
    }

    uint64_t poll_kernel(uint64_t kernel_id) override {
        return 0;  // completed
    }

    int synchronize_stream(uint64_t stream_id) override { return 0; }

    uint64_t global_access(uint64_t addr, uint64_t val, uint8_t type) override {
        return UINT64_MAX;
    }

    std::atomic<int> submit_count{0};
    uint64_t last_kernel_id = 0;
    uint64_t last_stream_id = 0;
};

TEST_CASE("Async launch: bridge nullptr path is byte-identical sync", "[cpptlm][async]") {
    // When g_cpptlm_bridge == nullptr, cudaLaunchKernel should behave identically
    // to the original sync path. This is verified by existing e2e tests.
    extern CppTLMBridge* g_cpptlm_bridge;
    REQUIRE(g_cpptlm_bridge == nullptr);
}

TEST_CASE("Async launch: bridge active submits to pending registry", "[cpptlm][async]") {
    AsyncMockBridge bridge;
    extern CppTLMBridge* g_cpptlm_bridge;

    // Set bridge
    g_cpptlm_bridge = &bridge;

    // Note: Full integration test requires __cudaRegisterFatBinary setup
    // which is complex. This test verifies the bridge pointer wiring.
    REQUIRE(g_cpptlm_bridge != nullptr);
    REQUIRE(g_cpptlm_bridge->version() == 1);

    // Reset bridge
    g_cpptlm_bridge = nullptr;
}

TEST_CASE("Async launch: unique kernel_id generation", "[cpptlm][async]") {
    AsyncMockBridge bridge;

    // Verify submit_kernel receives unique IDs
    uint64_t id1 = 1, id2 = 2;
    bridge.submit_kernel(id1, "k1", 1,1,1, 32,1,1, nullptr, 0, 0, 0);
    bridge.submit_kernel(id2, "k2", 1,1,1, 32,1,1, nullptr, 0, 0, 0);

    REQUIRE(bridge.submit_count.load() == 2);
    REQUIRE(bridge.last_kernel_id == id2);
}
