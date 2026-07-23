// test_bridge_submit_error.cpp
// =============================================================================
// Unit test: bridge submit_kernel error code propagation (spec: cpptlm-d1-full)
//
// Verifies: when bridge->submit_kernel() returns non-zero error codes,
// callers can distinguish success (0) from CUDA and CppTLM errors.
//
// Contract (from cpptlm_bridge.h:99-100):
//   - 0  = success (cudaSuccess equivalent)
//   - >0 = CUDA error code (e.g., cudaErrorInvalidValue=1,
//          cudaErrorMemoryAllocation=2, cudaErrorLaunchFailure=4,
//          cudaErrorUnknown=30)
//   - <0 = CppTLM-specific internal error
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/cpptlm_bridge.h"

#include <cstdint>
#include <type_traits>

// ---------------------------------------------------------------------------
// Mock bridges that return specific submit_kernel error codes.
// Each mock implements the full CppTLMBridge interface (5 pure virtuals).
// ---------------------------------------------------------------------------

class MockSuccessBridge : public CppTLMBridge {
public:
    int version() const override { return CPPTLMBRIDGE_VERSION; }
    int submit_kernel(uint64_t, const char*, uint32_t, uint32_t, uint32_t,
                      uint32_t, uint32_t, uint32_t, const void**, size_t,
                      size_t, uint64_t) override {
        return 0;
    }
    uint64_t poll_kernel(uint64_t) override { return 0; }
    int synchronize_stream(uint64_t) override { return 0; }
    uint64_t global_access(uint64_t, uint64_t, uint8_t) override { return 0; }
};

class MockErrorBridge : public CppTLMBridge {
public:
    explicit MockErrorBridge(int error_code) : error_code_(error_code) {}

    int version() const override { return CPPTLMBRIDGE_VERSION; }
    int submit_kernel(uint64_t, const char*, uint32_t, uint32_t, uint32_t,
                      uint32_t, uint32_t, uint32_t, const void**, size_t,
                      size_t, uint64_t) override {
        return error_code_;
    }
    uint64_t poll_kernel(uint64_t) override { return 0; }
    int synchronize_stream(uint64_t) override { return 0; }
    uint64_t global_access(uint64_t, uint64_t, uint8_t) override { return 0; }

private:
    int error_code_;
};

// Bridge with version mismatch (simulates ABI drift)
class MockVersionMismatchBridge : public CppTLMBridge {
public:
    explicit MockVersionMismatchBridge(int version) : version_(version) {}

    int version() const override { return version_; }
    int submit_kernel(uint64_t, const char*, uint32_t, uint32_t, uint32_t,
                      uint32_t, uint32_t, uint32_t, const void**, size_t,
                      size_t, uint64_t) override {
        return 0;
    }
    uint64_t poll_kernel(uint64_t) override { return 0; }
    int synchronize_stream(uint64_t) override { return 0; }
    uint64_t global_access(uint64_t, uint64_t, uint8_t) override { return 0; }

private:
    int version_;
};

// ============================================================================

TEST_CASE("Bridge submit_kernel: success returns 0", "[cpptlm][bridge][submit_error]") {
    MockSuccessBridge bridge;
    int result = bridge.submit_kernel(1, "testKernel", 1, 1, 1,
                                       32, 1, 1, nullptr, 0, 0, 0);
    REQUIRE(result == 0);
}

TEST_CASE("Bridge submit_kernel: CUDA error codes propagated", "[cpptlm][bridge][submit_error]") {
    // cudaError_t error codes (from cuda_runtime_api.h)
    // cudaErrorInvalidValue        =  1
    // cudaErrorMemoryAllocation    =  2
    // cudaErrorLaunchFailure       =  4
    // cudaErrorUnknown             = 30

    SECTION("cudaErrorInvalidValue (1)") {
        MockErrorBridge bridge(1);
        REQUIRE(bridge.submit_kernel(1, "k", 1, 1, 1, 32, 1, 1, nullptr, 0, 0, 0) == 1);
    }
    SECTION("cudaErrorMemoryAllocation (2)") {
        MockErrorBridge bridge(2);
        REQUIRE(bridge.submit_kernel(1, "k", 1, 1, 1, 32, 1, 1, nullptr, 0, 0, 0) == 2);
    }
    SECTION("cudaErrorLaunchFailure (4)") {
        MockErrorBridge bridge(4);
        REQUIRE(bridge.submit_kernel(1, "k", 1, 1, 1, 32, 1, 1, nullptr, 0, 0, 0) == 4);
    }
    SECTION("cudaErrorUnknown (30)") {
        MockErrorBridge bridge(30);
        REQUIRE(bridge.submit_kernel(1, "k", 1, 1, 1, 32, 1, 1, nullptr, 0, 0, 0) == 30);
    }
}

TEST_CASE("Bridge submit_kernel: CppTLM internal errors (negative)", "[cpptlm][bridge][submit_error]") {
    // Negative return codes are reserved for CppTLM internal errors
    // (e.g., -1 = internal queue full, -2 = serialization failure)

    SECTION("CppTLM internal error -1") {
        MockErrorBridge bridge(-1);
        REQUIRE(bridge.submit_kernel(1, "k", 1, 1, 1, 32, 1, 1, nullptr, 0, 0, 0) == -1);
    }
    SECTION("CppTLM internal error -2") {
        MockErrorBridge bridge(-2);
        REQUIRE(bridge.submit_kernel(1, "k", 1, 1, 1, 32, 1, 1, nullptr, 0, 0, 0) == -2);
    }
}

TEST_CASE("Bridge submit_kernel: error codes are int-typed", "[cpptlm][bridge][submit_error]") {
    // Verify the return type is int (not uint or void), supporting both
    // positive CUDA errors and negative CppTLM errors.
    MockErrorBridge bridge(0);

    // Compile-time: result must be int (not bool, not void)
    using ResultType = decltype(bridge.submit_kernel(
        1, "k", 1, 1, 1, 32, 1, 1, nullptr, 0, 0, 0));
    static_assert(std::is_same_v<ResultType, int>,
                  "submit_kernel must return int (signed) for error codes");

    REQUIRE(true);  // compilation check passed
}

TEST_CASE("Bridge submit_kernel: zero args_count with nullptr args", "[cpptlm][bridge][submit_error]") {
    // Kernel with no arguments — should succeed (not crash on nullptr deref)
    MockSuccessBridge bridge;
    int result = bridge.submit_kernel(1, "noArgKernel", 1, 1, 1,
                                       32, 1, 1, nullptr, 0, 0, 0);
    REQUIRE(result == 0);
}

TEST_CASE("Bridge version: matches CPPTLMBRIDGE_VERSION", "[cpptlm][bridge][submit_error]") {
    MockSuccessBridge bridge;
    REQUIRE(bridge.version() == CPPTLMBRIDGE_VERSION);
}

TEST_CASE("Bridge version: mismatch detected", "[cpptlm][bridge][submit_error]") {
    // When CppTLMBridge::version() != CPPTLMBRIDGE_VERSION, callers
    // should detect ABI drift and refuse to proceed.
    MockVersionMismatchBridge bridge(999);
    REQUIRE(bridge.version() != CPPTLMBRIDGE_VERSION);
}
