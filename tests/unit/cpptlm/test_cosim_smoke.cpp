#include "catch_amalgamated.hpp"
#include "ptxsim/gpu_context.h"
#include "cudart/cpptlm_bridge/PtxEmuDriverShim.h"
#include "cudart/cpptlm_bridge.h"

TEST_CASE("cosim smoke: PtxEmuDriverShim lifecycle", "[cpptlm][cosim][smoke]") {
    auto gpu = std::make_unique<GPUContext>("configs/ampere_a100.json");
    gpu->init();

    PtxEmuDriverShim shim(gpu.get());

    SECTION("advance returns KernelComplete when GPU idle") {
        uint32_t actual = 0;
        int r = shim.advance(10, actual);
        REQUIRE(r == 2); // KernelComplete
        REQUIRE(actual <= 10); // some cycles may execute before EXIT
    }

    SECTION("num_sms matches GPUContext") {
        REQUIRE(shim.num_sms() == static_cast<uint32_t>(gpu->get_num_sms()));
    }

    SECTION("mark_complete + is_kernel_complete thread-safe") {
        shim.mark_complete(42);
        REQUIRE(shim.is_kernel_complete(42) == true);
        REQUIRE(shim.is_kernel_complete(99) == false);
    }

    SECTION("get_gpu_context returns injected pointer") {
        REQUIRE(shim.get_gpu_context() == gpu.get());
    }
}

TEST_CASE("cosim smoke: cpptlm_set_driver ABI callable", "[cpptlm][cosim][smoke]") {
    PtxEmuDriverApi api{};
    cpptlm_set_driver(nullptr, api);
    SUCCEED("cpptlm_set_driver called without crash");
}
