#include "catch_amalgamated.hpp"
#include <cuda.h>
#include <vector>
#include <numeric>
#include <cstdio>
#include <cstring>
#include <cstdint>

// IMPORTANT: cuModuleLoad (cudart_sim.cpp:510) is a STUB that always returns success
// without registering kernels. We MUST use cuModuleLoadData (line 519) which goes
// through global_registry().insert() with real PTXIRLoader deserialization.
// The blob format expected: "PTXIR" (5 bytes) + LE u32 size + body.
//
// NOTE: PTX-EMU's libcudart.so does NOT export cuMemAlloc/cuMemcpy/cuMemFree.
// Scenarios requiring memory management (2.1, 2.4) are skipped in this build;
// cuMem* coverage is provided by integration_cuda_driver_api instead.
//
// PTX-EMU uses local error codes (per module_registry.h):
//   CUDA_ERROR_INVALID_VALUE = 1
//   CUDA_ERROR_INVALID_HANDLE = 2
//   CUDA_ERROR_NOT_FOUND = 3
//   CUDA_ERROR_INVALID_IMAGE = 4
//   CUDA_ERROR_INVALID_PTX = 5
// (NOT real CUDA's 1/2/500/200/218 values)

static std::vector<uint8_t> load_blob(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data(sz);
    if (fread(data.data(), 1, sz, f) != (size_t)sz) {
        fclose(f);
        return {};
    }
    fclose(f);
    return data;
}

TEST_CASE("Path 1C Scenario 2.2: duplicate module load", "[e2e][path_1C]") {
    CUmodule m1, m2;
    auto blob = load_blob("./vec_add.ptxir_blob");
    REQUIRE(!blob.empty());
    REQUIRE(cuModuleLoadData(&m1, blob.data()) == CUDA_SUCCESS);
    REQUIRE(cuModuleLoadData(&m2, blob.data()) == CUDA_SUCCESS);
    REQUIRE(m1 != m2);
    cuModuleUnload(m1); cuModuleUnload(m2);
}

TEST_CASE("Path 1C Scenario 2.3: kernel name not found", "[e2e][path_1C]") {
    CUmodule mod; CUfunction func;
    auto blob = load_blob("./vec_add.ptxir_blob");
    REQUIRE(!blob.empty());
    REQUIRE(cuModuleLoadData(&mod, blob.data()) == CUDA_SUCCESS);
    REQUIRE(cuModuleGetFunction(&func, mod, "nonexistent_kernel") == (CUresult)3);
    cuModuleUnload(mod);
}

TEST_CASE("Path 1C Scenario 2.5: cuModuleUnload invalidates func2name", "[e2e][path_1C]") {
    CUmodule mod; CUfunction func;
    auto blob = load_blob("./vec_add.ptxir_blob");
    REQUIRE(!blob.empty());
    REQUIRE(cuModuleLoadData(&mod, blob.data()) == CUDA_SUCCESS);
    REQUIRE(cuModuleGetFunction(&func, mod, "vec_add") == CUDA_SUCCESS);
    REQUIRE(cuModuleUnload(mod) == CUDA_SUCCESS);
    CUfunction func2;
    auto rc = cuModuleGetFunction(&func2, mod, "vec_add");
    REQUIRE(rc != CUDA_SUCCESS);
}

TEST_CASE("Path 1C Scenario 2.6: cuModuleLoadData null args", "[e2e][path_1C]") {
    CUmodule mod;
    REQUIRE(cuModuleLoadData(nullptr, nullptr) == (CUresult)1);
    REQUIRE(cuModuleLoadData(&mod, nullptr) == (CUresult)1);
}

TEST_CASE("Path 1C Scenario 2.7: cuModuleLoadData non-PTXIR magic", "[e2e][path_1C]") {
    CUmodule mod;
    uint8_t bad_blob[14] = {'W','R','O','N','G','_','M','A','G','I','C','!','!','!'};
    REQUIRE(cuModuleLoadData(&mod, bad_blob) == (CUresult)4);
}
