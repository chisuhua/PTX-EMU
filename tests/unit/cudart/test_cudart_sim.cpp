// cudaLaunchKernel unit tests require PTX context setup (func2name map,
// g_ptx_interpreter) that is only available after __cudaRegisterFatBinary.
// Those tests belong in tests/e2e/ or tests/integration/.
// Here we test cudaStreamSynchronize which operates independently.

#include "catch_amalgamated.hpp"
#include "cudart/cudart_intrinsics.h"

extern "C" {
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
}

TEST_CASE("cudaStreamSynchronize on default stream returns success",
          "[cudart][sync]") {
    REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);
}

TEST_CASE("cudaStreamSynchronize with no pending launch",
          "[cudart][sync]") {
    REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);
}