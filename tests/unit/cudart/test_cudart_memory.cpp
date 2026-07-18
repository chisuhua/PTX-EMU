// test_cudart_memory.cpp
// =============================================================================
// Unit test: CUDA Runtime Memory API (cudaMalloc, cudaFree, cudaMemcpy, cudaMemset)
//
// Tests the fake libcudart.so memory interception layer in cudart_sim.cpp.
// CudaDriver singleton requires SimpleMemory initialization before memory
// operations can succeed. Each test case sets up a fresh SimpleMemory instance
// via a CudaDriverInit fixture.
//
// Uses project-internal cudart_intrinsics.h instead of <cuda_runtime.h>
// because this file is compiled with g++ (not nvcc).
//
// Ref: ADR-0010 (add-cudart-unit-test-coverage)
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/cuda_driver.h"
#include "cudart/cudart_intrinsics.h"
#include "memory/simple_memory.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

// CUDA runtime entry points (C linkage, defined in cudart_sim.cpp).
extern "C" {
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
}

// ---------------------------------------------------------------------------
// Fixture: initialize CudaDriver with a SimpleMemory instance per test case.
// Without this, CudaDriver::get_global_pool() returns nullptr and all memory
// operations fail.
// ---------------------------------------------------------------------------
struct CudaDriverInit {
    static constexpr size_t GLOBAL_MEM_SIZE = 16 * 1024 * 1024; // 16 MiB

    CudaDriverInit() {
        simple_mem = std::make_unique<SimpleMemory>(GLOBAL_MEM_SIZE);
        CudaDriver::instance().set_simple_memory(simple_mem.get());
    }

    ~CudaDriverInit() {
        // CudaDriver has no reset() API. Subsequent tests reuse the same
        // singleton with the same SimpleMemory pool. This is safe because
        // each test allocates/frees within the pool independently.
    }

    std::unique_ptr<SimpleMemory> simple_mem;
};

// =============================================================================
// cudaMalloc
// =============================================================================

TEST_CASE_METHOD(CudaDriverInit, "cudaMalloc allocates device memory",
                 "[cudart][memory][malloc]") {
    void *devPtr = nullptr;
    REQUIRE(cudaMalloc(&devPtr, 1024) == cudaSuccess);
    REQUIRE(devPtr != nullptr);
    REQUIRE(cudaFree(devPtr) == cudaSuccess);
}

TEST_CASE_METHOD(CudaDriverInit, "cudaMalloc various sizes",
                 "[cudart][memory][malloc]") {
    const size_t sizes[] = {1, 256, 4096, 65536, 1048576};
    for (size_t sz : sizes) {
        void *devPtr = nullptr;
        REQUIRE(cudaMalloc(&devPtr, sz) == cudaSuccess);
        REQUIRE(devPtr != nullptr);
        REQUIRE(cudaFree(devPtr) == cudaSuccess);
    }
}

TEST_CASE_METHOD(CudaDriverInit, "cudaMalloc null devPtr returns InvalidValue",
                 "[cudart][memory][malloc]") {
    REQUIRE(cudaMalloc(nullptr, 1024) == cudaErrorInvalidValue);
}

// =============================================================================
// cudaFree
// =============================================================================

TEST_CASE_METHOD(CudaDriverInit, "cudaFree valid pointer returns Success",
                 "[cudart][memory][free]") {
    void *devPtr = nullptr;
    REQUIRE(cudaMalloc(&devPtr, 512) == cudaSuccess);
    REQUIRE(cudaFree(devPtr) == cudaSuccess);
}

TEST_CASE_METHOD(CudaDriverInit, "cudaFree invalid pointer returns InvalidValue",
                 "[cudart][memory][free]") {
    // A pointer that was never allocated by CudaDriver should fail.
    void *bogus = reinterpret_cast<void *>(0xDEADBEEF);
    REQUIRE(cudaFree(bogus) == cudaErrorInvalidValue);
}

TEST_CASE_METHOD(CudaDriverInit, "cudaFree nullptr returns Success",
                 "[cudart][memory][free]") {
    // Per CUDA spec: cudaFree(nullptr) is a no-op returning cudaSuccess.
    // CudaDriver::free(nullptr) returns Success.
    REQUIRE(cudaFree(nullptr) == cudaSuccess);
}

// =============================================================================
// cudaMemcpy HostToDevice / DeviceToHost
// =============================================================================

TEST_CASE_METHOD(CudaDriverInit, "cudaMemcpy H2D and D2H round-trip",
                 "[cudart][memory][memcpy]") {
    constexpr size_t N = 256;
    std::vector<uint8_t> host_src(N);
    std::vector<uint8_t> host_dst(N, 0);
    for (size_t i = 0; i < N; ++i) {
        host_src[i] = static_cast<uint8_t>(i);
    }

    void *devPtr = nullptr;
    REQUIRE(cudaMalloc(&devPtr, N) == cudaSuccess);

    // Host → Device
    REQUIRE(cudaMemcpy(devPtr, host_src.data(), N, cudaMemcpyHostToDevice) ==
            cudaSuccess);

    // Device → Host
    REQUIRE(cudaMemcpy(host_dst.data(), devPtr, N, cudaMemcpyDeviceToHost) ==
            cudaSuccess);

    // Verify data integrity
    for (size_t i = 0; i < N; ++i) {
        REQUIRE(host_dst[i] == host_src[i]);
    }

    REQUIRE(cudaFree(devPtr) == cudaSuccess);
}

TEST_CASE_METHOD(CudaDriverInit, "cudaMemcpy H2D partial write and D2H verify",
                 "[cudart][memory][memcpy]") {
    constexpr size_t N = 1024;
    std::vector<uint32_t> host_src(N);
    std::vector<uint32_t> host_dst(N, 0);
    for (size_t i = 0; i < N; ++i) {
        host_src[i] = static_cast<uint32_t>(i * 7 + 3);
    }

    void *devPtr = nullptr;
    REQUIRE(cudaMalloc(&devPtr, N * sizeof(uint32_t)) == cudaSuccess);

    REQUIRE(cudaMemcpy(devPtr, host_src.data(), N * sizeof(uint32_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(host_dst.data(), devPtr, N * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(host_dst[i] == host_src[i]);
    }

    REQUIRE(cudaFree(devPtr) == cudaSuccess);
}

TEST_CASE_METHOD(CudaDriverInit, "cudaMemcpy null dst returns InvalidValue",
                 "[cudart][memory][memcpy]") {
    uint8_t buf[16] = {};
    void *devPtr = nullptr;
    REQUIRE(cudaMalloc(&devPtr, 16) == cudaSuccess);

    REQUIRE(cudaMemcpy(nullptr, buf, 16, cudaMemcpyHostToDevice) ==
            cudaErrorInvalidValue);

    REQUIRE(cudaFree(devPtr) == cudaSuccess);
}

TEST_CASE_METHOD(CudaDriverInit, "cudaMemcpy null src returns InvalidValue",
                 "[cudart][memory][memcpy]") {
    void *devPtr = nullptr;
    REQUIRE(cudaMalloc(&devPtr, 16) == cudaSuccess);

    REQUIRE(cudaMemcpy(devPtr, nullptr, 16, cudaMemcpyHostToDevice) ==
            cudaErrorInvalidValue);

    REQUIRE(cudaFree(devPtr) == cudaSuccess);
}

TEST_CASE_METHOD(CudaDriverInit, "cudaMemcpy count=0 returns InvalidValue",
                 "[cudart][memory][memcpy]") {
    uint8_t buf[1] = {};
    void *devPtr = nullptr;
    REQUIRE(cudaMalloc(&devPtr, 16) == cudaSuccess);

    REQUIRE(cudaMemcpy(devPtr, buf, 0, cudaMemcpyHostToDevice) ==
            cudaErrorInvalidValue);

    REQUIRE(cudaFree(devPtr) == cudaSuccess);
}

// =============================================================================
// cudaMemset
// =============================================================================

TEST_CASE_METHOD(CudaDriverInit, "cudaMemset sets device memory and verify via D2H",
                 "[cudart][memory][memset]") {
    constexpr size_t N = 128;
    void *devPtr = nullptr;
    REQUIRE(cudaMalloc(&devPtr, N) == cudaSuccess);

    // Set all bytes to 0xAB
    REQUIRE(cudaMemset(devPtr, 0xAB, N) == cudaSuccess);

    // Read back and verify
    std::vector<uint8_t> host_buf(N, 0);
    REQUIRE(cudaMemcpy(host_buf.data(), devPtr, N, cudaMemcpyDeviceToHost) ==
            cudaSuccess);

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(host_buf[i] == 0xAB);
    }

    REQUIRE(cudaFree(devPtr) == cudaSuccess);
}

TEST_CASE_METHOD(CudaDriverInit, "cudaMemset null devPtr returns InvalidValue",
                 "[cudart][memory][memset]") {
    REQUIRE(cudaMemset(nullptr, 0xFF, 64) == cudaErrorInvalidValue);
}

TEST_CASE_METHOD(CudaDriverInit, "cudaMemset with zero value",
                 "[cudart][memory][memset]") {
    constexpr size_t N = 64;
    void *devPtr = nullptr;
    REQUIRE(cudaMalloc(&devPtr, N) == cudaSuccess);

    // First set to non-zero, then zero it
    REQUIRE(cudaMemset(devPtr, 0xCC, N) == cudaSuccess);
    REQUIRE(cudaMemset(devPtr, 0x00, N) == cudaSuccess);

    std::vector<uint8_t> host_buf(N, 0xFF);
    REQUIRE(cudaMemcpy(host_buf.data(), devPtr, N, cudaMemcpyDeviceToHost) ==
            cudaSuccess);

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(host_buf[i] == 0x00);
    }

    REQUIRE(cudaFree(devPtr) == cudaSuccess);
}
