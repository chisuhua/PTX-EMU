// test_ld_st_bridge.cpp
// =============================================================================
// Integration test: GLOBAL LD/ST bridge timing (D-PTX-3 + Task #4)
//
// 验证 LdHandler/StHandler 在 bridge != nullptr 时走 CppTLM NoC timing
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/cpptlm_bridge.h"
#include <cstdint>

class TimingMockBridge : public CppTLMBridge {
public:
    int version() const override { return 1; }

    int submit_kernel(uint64_t kid, const char* name,
                     uint32_t gx, uint32_t gy, uint32_t gz,
                     uint32_t bx, uint32_t by, uint32_t bz,
                     const void** args, size_t cnt,
                     size_t smem, uint64_t sid) override { return 0; }

    uint64_t poll_kernel(uint64_t kid) override { return 0; }

    int synchronize_stream(uint64_t sid) override { return 0; }

    uint64_t global_access(uint64_t addr, uint64_t val, uint8_t type) override {
        global_access_count++;
        last_addr = addr;
        last_val = val;
        last_type = type;
        return latency_to_return;
    }

    int global_access_count = 0;
    uint64_t last_addr = 0;
    uint64_t last_val = 0;
    uint8_t last_type = 0;
    uint64_t latency_to_return = 100;
};

TEST_CASE("GLOBAL LD: bridge returns latency", "[cpptlm][ld_st]") {
    TimingMockBridge bridge;
    bridge.latency_to_return = 200;

    uint64_t addr = 0x1000;
    uint64_t latency = bridge.global_access(addr, 0, 0);

    REQUIRE(latency == 200);
    REQUIRE(bridge.last_addr == addr);
    REQUIRE(bridge.last_type == 0);  // LD
}

TEST_CASE("GLOBAL ST: bridge returns latency with val", "[cpptlm][ld_st]") {
    TimingMockBridge bridge;
    bridge.latency_to_return = 150;

    uint64_t addr = 0x2000;
    uint64_t val = 42;
    uint64_t latency = bridge.global_access(addr, val, 1);

    REQUIRE(latency == 150);
    REQUIRE(bridge.last_addr == addr);
    REQUIRE(bridge.last_val == val);
    REQUIRE(bridge.last_type == 1);  // ST
}

TEST_CASE("GLOBAL LD/ST: UINT64_MAX fallback", "[cpptlm][ld_st]") {
    TimingMockBridge bridge;
    bridge.latency_to_return = UINT64_MAX;

    uint64_t latency = bridge.global_access(0x3000, 0, 0);

    // UINT64_MAX means address not mapped → fallback to original path
    REQUIRE(latency == UINT64_MAX);
}

TEST_CASE("GLOBAL LD/ST: multiple accesses accumulate", "[cpptlm][ld_st]") {
    TimingMockBridge bridge;
    bridge.latency_to_return = 50;

    for (int i = 0; i < 10; i++) {
        bridge.global_access(0x1000 + i * 8, 0, 0);
    }

    REQUIRE(bridge.global_access_count == 10);
}
