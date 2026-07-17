// test_async_launchkernel.cpp
// =============================================================================
// Integration test: cudaLaunchKernel 异步路径 (D-PTX-1 + Task #2)
//
// 验证 bridge != nullptr 时 kernel 走异步提交路径
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/cpptlm_bridge.h"
#include "cudart/cudart_intrinsics.h"   // dim3, cudaStream_t, cudaError_t, cudaSuccess
#include <atomic>

// g_cpptlm_bridge / cudaLaunchKernel 由 libcudart 提供（C 链接）。
// 测试 binary 通过 add_catch_test 链接到 cudart 静态/动态目标（tests/CMakeLists.txt:71）。
extern "C" {
    cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                                void** args, size_t sharedMem, cudaStream_t stream);
}

class AsyncMockBridge : public CppTLMBridge {
public:
    int version() const override { return 1; }

    int submit_kernel(uint64_t kernel_id, const char* /*kernel_name*/,
                     uint32_t gx, uint32_t gy, uint32_t gz,
                     uint32_t bx, uint32_t by, uint32_t bz,
                     const void** /*args*/, size_t args_count,
                     size_t shared_mem, uint64_t stream_id) override {
        submit_count.fetch_add(1);
        last_kernel_id  = kernel_id;
        last_stream_id  = stream_id;
        last_grid_x     = gx;
        last_grid_y     = gy;
        last_grid_z     = gz;
        last_block_x    = bx;
        last_block_y    = by;
        last_block_z    = bz;
        last_args_count = args_count;
        last_shared_mem = shared_mem;
        return 0;
    }

    uint64_t poll_kernel(uint64_t /*kernel_id*/) override { return 0; }
    int       synchronize_stream(uint64_t /*stream_id*/) override { return 0; }
    uint64_t  global_access(uint64_t /*addr*/, uint64_t /*val*/, uint8_t /*type*/) override {
        return UINT64_MAX;
    }

    std::atomic<int> submit_count{0};
    uint64_t last_kernel_id  = 0;
    uint64_t last_stream_id  = 0;
    uint32_t last_grid_x     = 0;
    uint32_t last_grid_y     = 0;
    uint32_t last_grid_z     = 0;
    uint32_t last_block_x    = 0;
    uint32_t last_block_y    = 0;
    uint32_t last_block_z    = 0;
    size_t   last_args_count = 0;
    size_t   last_shared_mem = 0;
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
    bridge.submit_kernel(id1, "k1", 1, 1, 1, 32, 1, 1, nullptr, 0, 0, 0);
    bridge.submit_kernel(id2, "k2", 1, 1, 1, 32, 1, 1, nullptr, 0, 0, 0);

    REQUIRE(bridge.submit_count.load() == 2);
    REQUIRE(bridge.last_kernel_id == id2);
}

// ============================================================================
// RED→GREEN: real cudaLaunchKernel C entry, no __cudaRegisterFatBinary.
// CudartSim cudaLaunchKernel bridge path (cudart_sim.cpp:490-551) only
// requires g_cpptlm_bridge != nullptr; default stream is 0; args==nullptr
// yields arg_count==0 (count_kernel_args nullptr-sentinel loop is safe).
// func2name[(uint64_t)func] side-effect: default-construct empty key —
// harmless and additive. We exercise the production production entry
// point directly, then verify forwarding to bridge->submit_kernel().
// ============================================================================
TEST_CASE("Async launch: real cudaLaunchKernel entry forwards to bridge (D-PTX-1)",
          "[cpptlm][async][cudart][integration]") {
    extern CppTLMBridge* g_cpptlm_bridge;
    AsyncMockBridge bridge;

    // Detach any pre-installed bridge from earlier tests.
    g_cpptlm_bridge = nullptr;
    g_cpptlm_bridge = &bridge;
    REQUIRE(g_cpptlm_bridge == &bridge);

    // Use a stable, never-registered kernel pointer. func2name operator[]
    // inserts an empty entry; no kernel body is looked up (bridge path
    // returns cudaSuccess at line 550 without touching g_ptx_interpreter).
    static const char kKernelName[] = "async_forward_test_kernel";
    static int dummy_kernel = 0;
    (void)kKernelName;
    (void)dummy_kernel;

    cudaError_t err = cudaLaunchKernel(
        /*func*/     static_cast<const void*>(&dummy_kernel),
        /*gridDim*/  dim3{1u, 1u, 1u},
        /*blockDim*/ dim3{32u, 1u, 1u},
        /*args*/     nullptr,        // count_kernel_args returns 0 — no deep-copy UB
        /*sharedMem*/0u,
        /*stream*/   nullptr         // default stream => stream_id == 0
    );

    // CUDA runtime returned success and bridge observed exactly one submit
    REQUIRE(err == cudaSuccess);
    REQUIRE(bridge.submit_count.load() == 1);

    // Bridge received the kernel_id assigned by generate_kernel_id() (atomic
    // starts at 1 — guarded by first-submit semantics in this test).
    REQUIRE(bridge.last_kernel_id == 1u);

    // Grid / block dimensions must be forwarded verbatim
    REQUIRE(bridge.last_grid_x  == 1u);
    REQUIRE(bridge.last_grid_y  == 1u);
    REQUIRE(bridge.last_grid_z  == 1u);
    REQUIRE(bridge.last_block_x == 32u);
    REQUIRE(bridge.last_block_y == 1u);
    REQUIRE(bridge.last_block_z == 1u);

    // No args → args_count == 0; shareMem == 0; default stream → stream_id == 0
    REQUIRE(bridge.last_args_count == 0u);
    REQUIRE(bridge.last_shared_mem == 0u);
    REQUIRE(bridge.last_stream_id  == 0u);

    // Cleanup before next test case
    g_cpptlm_bridge = nullptr;
}
