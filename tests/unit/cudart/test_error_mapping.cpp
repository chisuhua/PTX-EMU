#include "catch_amalgamated.hpp"
#include "cudart/cudart_intrinsics.h"
#include "cudart/module_registry.h"
#include <cstring>

extern "C" {
    CUresult cuModuleGetFunction(CUfunction*, CUmodule, const char*);
    CUresult cuModuleUnload(CUmodule);
    CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned,
                           unsigned, unsigned, unsigned,
                           unsigned, CUstream, void**, void**);
}

namespace {

TEST_CASE("cuLaunchKernel: NULL func returns INVALID_VALUE", "[cudart][driver_api]") {
    void* params[1] = {nullptr};
    CUresult rc = cuLaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, params, nullptr);
    REQUIRE(rc == ptxemu::cudart::CUDA_ERROR_INVALID_VALUE);
}

TEST_CASE("cuLaunchKernel: NULL params returns INVALID_VALUE", "[cudart][driver_api]") {
    CUresult rc = cuLaunchKernel(reinterpret_cast<CUfunction>(0x1), 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr);
    REQUIRE(rc == ptxemu::cudart::CUDA_ERROR_INVALID_VALUE);
}

TEST_CASE("cuModuleUnload: NULL handle returns INVALID_VALUE", "[cudart][driver_api]") {
    CUresult rc = cuModuleUnload(nullptr);
    REQUIRE(rc == ptxemu::cudart::CUDA_ERROR_INVALID_VALUE);
}

TEST_CASE("cuModuleUnload: invalid handle returns INVALID_HANDLE", "[cudart][driver_api]") {
    CUresult rc = cuModuleUnload(reinterpret_cast<CUmodule>(0xDEADBEEF));
    REQUIRE(rc == ptxemu::cudart::CUDA_ERROR_INVALID_HANDLE);
}

TEST_CASE("cuModuleGetFunction: NULL handle returns INVALID_VALUE", "[cudart][driver_api]") {
    CUfunction f = nullptr;
    CUresult rc = cuModuleGetFunction(&f, nullptr, "foo");
    REQUIRE(rc == ptxemu::cudart::CUDA_ERROR_INVALID_VALUE);
}

TEST_CASE("cuModuleGetFunction: NULL module returns INVALID_VALUE", "[cudart][driver_api]") {
    CUfunction f = nullptr;
    CUresult rc = cuModuleGetFunction(&f, reinterpret_cast<CUmodule>(1), nullptr);
    REQUIRE(rc == ptxemu::cudart::CUDA_ERROR_INVALID_VALUE);
}

TEST_CASE("cuModuleGetFunction: unknown module returns INVALID_HANDLE", "[cudart][driver_api]") {
    CUfunction f = nullptr;
    CUresult rc = cuModuleGetFunction(&f, reinterpret_cast<CUmodule>(0xCAFEBABE), "foo");
    REQUIRE(rc == ptxemu::cudart::CUDA_ERROR_INVALID_HANDLE);
}

TEST_CASE("cuLaunchKernel: stale function handle returns INVALID_HANDLE", "[cudart][driver_api]") {
    auto& reg = ptxemu::cudart::global_registry();
    uint8_t image[16] = {0};
    CUmodule mod = nullptr;
    REQUIRE(reg.insert(image, sizeof(image), &mod) == CUDA_SUCCESS);

    CUfunction func = nullptr;
    CUresult rc = cuModuleGetFunction(&func, mod, "mykernel");
    REQUIRE(rc == CUDA_SUCCESS);
    REQUIRE(func != nullptr);

    cuModuleUnload(mod);

    void* params[1] = {nullptr};
    rc = cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, nullptr, params, nullptr);
    REQUIRE(rc == ptxemu::cudart::CUDA_ERROR_INVALID_HANDLE);
}

}  // namespace
