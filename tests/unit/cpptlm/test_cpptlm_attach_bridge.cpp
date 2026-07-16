// test_cpptlm_attach_bridge.cpp
// =============================================================================
// Unit test: cpptlm_attach_bridge / cpptlm_detach_bridge ABI entry points (B1)
//
// Per Metis second-pass review (B1): cpptlm_bridge.h:161,168 declare two
// extern "C" PTXEMU_BRIDGE_API functions but no TU defined them — link error
// when CppTLM's libcpptlm_cudart.so calls them.
//
// This test verifies:
//   1. cpptlm_attach_bridge(mock) sets g_cpptlm_bridge == mock
//   2. cpptlm_attach_bridge is idempotent (second call overwrites first)
//   3. cpptlm_detach_bridge() resets g_cpptlm_bridge to nullptr
//   4. cpptlm_detach_bridge is safe to call when already nullptr (idempotent)
//
// Ref: docs/adr/0021-cpptlm-d1-full-integration.md §D-PTX-1
//      docs/dev-process/lessons-learned.md §1 (ABI declaration + impl paired)
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/cpptlm_bridge.h"

// Mock Bridge — minimal concrete implementation for attach/detach testing.
class MockBridgeForAttach : public CppTLMBridge {
public:
    int version() const override { return CPPTLMBRIDGE_VERSION; }
    int submit_kernel(uint64_t, const char*, uint32_t, uint32_t, uint32_t,
                      uint32_t, uint32_t, uint32_t, const void**, size_t,
                      size_t, uint64_t) override { return 0; }
    uint64_t poll_kernel(uint64_t) override { return 0; }
    int synchronize_stream(uint64_t) override { return 0; }
    uint64_t global_access(uint64_t, uint64_t, uint8_t) override { return 0; }
};

// ABI entry points declared in cpptlm_bridge.h:161,168 — must be defined in
// src/cudart/cudart_sim.cpp (same TU as g_cpptlm_bridge per ADR-0021 D-PTX-1).
extern "C" {
void cpptlm_attach_bridge(CppTLMBridge* bridge);
void cpptlm_detach_bridge();
}

// Global pointer defined in cudart_sim.cpp — read here for verification.
extern CppTLMBridge* g_cpptlm_bridge;

TEST_CASE("cpptlm_attach_bridge sets g_cpptlm_bridge", "[cpptlm][bridge][attach]") {
    // Ensure clean state before test (test isolation).
    cpptlm_detach_bridge();
    REQUIRE(g_cpptlm_bridge == nullptr);

    MockBridgeForAttach mock;
    cpptlm_attach_bridge(&mock);
    REQUIRE(g_cpptlm_bridge == &mock);

    // Cleanup — leave g_cpptlm_bridge nullptr for subsequent tests.
    cpptlm_detach_bridge();
    REQUIRE(g_cpptlm_bridge == nullptr);
}

TEST_CASE("cpptlm_attach_bridge is idempotent (overwrites)", "[cpptlm][bridge][attach]") {
    cpptlm_detach_bridge();
    REQUIRE(g_cpptlm_bridge == nullptr);

    MockBridgeForAttach mock1;
    MockBridgeForAttach mock2;
    cpptlm_attach_bridge(&mock1);
    REQUIRE(g_cpptlm_bridge == &mock1);

    // Second attach must overwrite the first (no-op guard would be a bug).
    cpptlm_attach_bridge(&mock2);
    REQUIRE(g_cpptlm_bridge == &mock2);
    REQUIRE(g_cpptlm_bridge != &mock1);

    cpptlm_detach_bridge();
    REQUIRE(g_cpptlm_bridge == nullptr);
}

TEST_CASE("cpptlm_detach_bridge is safe when already nullptr", "[cpptlm][bridge][attach]") {
    cpptlm_detach_bridge();
    REQUIRE(g_cpptlm_bridge == nullptr);

    // Calling detach when already nullptr must not crash (idempotent).
    cpptlm_detach_bridge();
    REQUIRE(g_cpptlm_bridge == nullptr);

    // Double-detach after a real attach also safe.
    MockBridgeForAttach mock;
    cpptlm_attach_bridge(&mock);
    cpptlm_detach_bridge();
    cpptlm_detach_bridge();  // second detach — must be no-op
    REQUIRE(g_cpptlm_bridge == nullptr);
}
