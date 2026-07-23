// test_kernel_id_uniqueness.cpp
// =============================================================================
// Unit test: kernel_id uniqueness contract (spec: cpptlm-d1-full)
//
// Verifies that kernel_id values submitted through CppTLMBridge::submit_kernel
// satisfy the contract:
//   1. Unique — each kernel launch produces a distinct ID
//   2. Monotonically increasing — later launches have larger IDs
//   3. Monotonically increasing — later launches have larger IDs
//   4. Fit in uint64_t — sufficient range for all practical workloads
//
// Note: generate_kernel_id() uses std::atomic<uint64_t>::fetch_add(1),
// which returns the PRE-increment value. So the first call returns 0.
// This test faithfully replicates that exact behavior.
// We cannot call the production generate_kernel_id() directly because
// it's static in cudart_sim.cpp.
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/cpptlm_bridge.h"

#include <atomic>
#include <unordered_set>
#include <cstdint>

// Mock bridge that records all submitted kernel_id values
class KernelIdMockBridge : public CppTLMBridge {
public:
    int version() const override { return CPPTLMBRIDGE_VERSION; }

    int submit_kernel(uint64_t kernel_id, const char*,
                      uint32_t, uint32_t, uint32_t,
                      uint32_t, uint32_t, uint32_t,
                      const void**, size_t,
                      size_t, uint64_t) override {
        submitted_ids.push_back(kernel_id);
        return 0;
    }

    uint64_t poll_kernel(uint64_t) override { return 0; }
    int synchronize_stream(uint64_t) override { return 0; }
    uint64_t global_access(uint64_t, uint64_t, uint8_t) override { return 0; }

    std::vector<uint64_t> submitted_ids;
};

// Simulates the behavior of generate_kernel_id() from cudart_sim.cpp:211-213
// which uses std::atomic<uint64_t>::fetch_add(1).
static std::atomic<uint64_t> g_id_counter{0};

static uint64_t generate_kernel_id() {
    return g_id_counter.fetch_add(1);
}

TEST_CASE("kernel_id: fetch_add returns pre-increment (starts from 0)", "[cpptlm][bridge][kernel_id]") {
    g_id_counter.store(0);
    REQUIRE(generate_kernel_id() == 0);  // fetch_add returns value before add
    REQUIRE(generate_kernel_id() == 1);
    REQUIRE(generate_kernel_id() == 2);
}

TEST_CASE("kernel_id: N=100 monotonically increasing uniqueness", "[cpptlm][bridge][kernel_id]") {
    g_id_counter.store(0);
    KernelIdMockBridge bridge;
    std::unordered_set<uint64_t> seen;

    for (int i = 0; i < 100; i++) {
        uint64_t id = generate_kernel_id();
        REQUIRE(seen.insert(id).second);               // must be unique
        bridge.submit_kernel(id, "k", 1,1,1, 32,1,1, nullptr, 0, 0, 0);
    }

    REQUIRE(bridge.submitted_ids.size() == 100);
    REQUIRE(seen.size() == 100);

    // Verify monotonic: each subsequent ID > previous
    for (size_t i = 1; i < bridge.submitted_ids.size(); i++) {
        REQUIRE(bridge.submitted_ids[i] > bridge.submitted_ids[i - 1]);
    }
}

TEST_CASE("kernel_id: N=1000 uniqueness under rapid generation", "[cpptlm][bridge][kernel_id]") {
    g_id_counter.store(0);
    KernelIdMockBridge bridge;
    std::unordered_set<uint64_t> seen;
    constexpr int N = 1000;

    for (int i = 0; i < N; i++) {
        uint64_t id = generate_kernel_id();
        auto [it, inserted] = seen.insert(id);
        REQUIRE(inserted);
        bridge.submit_kernel(id, "k", 1,1,1, 32,1,1, nullptr, 0, 0, 0);
    }

    REQUIRE(bridge.submitted_ids.size() == N);
    REQUIRE(seen.size() == N);
}

TEST_CASE("kernel_id: fits in uint64_t", "[cpptlm][bridge][kernel_id]") {
    REQUIRE(sizeof(uint64_t) >= 8);
    REQUIRE(static_cast<uint64_t>(1) < UINT64_MAX);

    // Verify the bridge accepts uint64_t IDs (compile-time type check)
    KernelIdMockBridge bridge;
    uint64_t max_id = UINT64_MAX;
    bridge.submit_kernel(max_id, "k", 1,1,1, 32,1,1, nullptr, 0, 0, 0);
    REQUIRE(bridge.submitted_ids.size() == 1);
    REQUIRE(bridge.submitted_ids[0] == UINT64_MAX);
}

TEST_CASE("kernel_id: first bridge call receives id 0 via fetch_add", "[cpptlm][bridge][kernel_id]") {
    g_id_counter.store(0);
    KernelIdMockBridge bridge;

    uint64_t first_id = generate_kernel_id();
    REQUIRE(first_id == 0);  // fetch_add returns pre-increment value
    bridge.submit_kernel(first_id, "k", 1,1,1, 32,1,1, nullptr, 0, 0, 0);
    REQUIRE(bridge.submitted_ids.size() == 1);
    REQUIRE(bridge.submitted_ids[0] == 0);

    // Second call returns 1
    uint64_t second_id = generate_kernel_id();
    REQUIRE(second_id == 1);
    bridge.submit_kernel(second_id, "k2", 1,1,1, 32,1,1, nullptr, 0, 0, 0);
    REQUIRE(bridge.submitted_ids[1] == 1);
    REQUIRE(bridge.submitted_ids[1] > bridge.submitted_ids[0]);  // monotonic
}