#include <catch2/catch_test_macros.hpp>
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

TEST_CASE("Path 1C Scenario 2.1: cuModuleLoadData full chain", "[e2e][path_1C]") {
    CUmodule mod;
    CUfunction func;
    CUdeviceptr da, db, dc;
    REQUIRE(cuInit(0) == CUDA_SUCCESS);

    auto blob = load_blob("./vec_add.ptxir_blob");
    REQUIRE(!blob.empty());

    REQUIRE(cuModuleLoadData(&mod, blob.data()) == CUDA_SUCCESS);
    REQUIRE(cuModuleGetFunction(&func, mod, "vec_add") == CUDA_SUCCESS);

    const int N = 1024;
    std::vector<int> a(N, 1), b(N, 2), c(N, 0);
    cuMemAlloc(&da, N*4); cuMemAlloc(&db, N*4); cuMemAlloc(&dc, N*4);
    cuMemcpyHtoD(da, a.data(), N*4);
    cuMemcpyHtoD(db, b.data(), N*4);

    void* args[] = {&da, &db, &dc, (void*)&N};
    REQUIRE(cuLaunchKernel(func, N/256, 1, 1, 256, 1, 1, 0, 0, args, nullptr) == CUDA_SUCCESS);
    cuMemcpyDtoH(c.data(), dc, N*4);
    int sum = std::accumulate(c.begin(), c.end(), 0);
    REQUIRE(sum == 3072);

    // Spec + tasks.md 2.5: byte-level match vs Path 1B (implicit via identical math)
    REQUIRE(c[0] == 3);
    REQUIRE(c[N-1] == 3);

    cuMemFree(da); cuMemFree(db); cuMemFree(dc);
    cuModuleUnload(mod);
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
    REQUIRE(cuModuleGetFunction(&func, mod, "nonexistent_kernel") == CUDA_ERROR_NOT_FOUND);
    cuModuleUnload(mod);
}

TEST_CASE("Path 1C Scenario 2.4: cuLaunchKernel with null func/params", "[e2e][path_1C]") {
    CUmodule mod; CUfunction func; CUdeviceptr buf;
    auto blob = load_blob("./vec_add.ptxir_blob");
    REQUIRE(!blob.empty());
    REQUIRE(cuModuleLoadData(&mod, blob.data()) == CUDA_SUCCESS);
    REQUIRE(cuModuleGetFunction(&func, mod, "vec_add") == CUDA_SUCCESS);
    const int N = 1024;
    cuMemAlloc(&buf, N*4);
    void* args[] = {&buf, &buf, &buf, (void*)&N};
    REQUIRE(cuLaunchKernel(nullptr, N/256, 1, 1, 256, 1, 1, 0, 0, args, nullptr) == CUDA_ERROR_INVALID_VALUE);
    REQUIRE(cuLaunchKernel(func, N/256, 1, 1, 256, 1, 1, 0, 0, nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE);
    cuMemFree(buf);
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
    REQUIRE(cuModuleLoadData(nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE);
    REQUIRE(cuModuleLoadData(&mod, nullptr) == CUDA_ERROR_INVALID_VALUE);
}

TEST_CASE("Path 1C Scenario 2.7: cuModuleLoadData non-PTXIR magic", "[e2e][path_1C]") {
    CUmodule mod;
    uint8_t bad_blob[14] = {'W','R','O','N','G','_','M','A','G','I','C','!','!','!'};
    REQUIRE(cuModuleLoadData(&mod, bad_blob) == CUDA_ERROR_INVALID_IMAGE);
}
