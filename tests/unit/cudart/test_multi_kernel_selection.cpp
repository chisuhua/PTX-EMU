#include "catch_amalgamated.hpp"
#include "cudart/cpptlm_module.h"
#include "ptx_ir/ptxir_format.h"

// Phase 12.4: this test is a STRUCTURAL PLACEHOLDER.
// Real validation requires a multi-entry PTXIR fixture, which requires
// v2 writer (out of Phase 12.4 scope; deferred to Phase 12.5).
// For now, verify that the KernelEntry struct and ManifestSection.kernels
// field are accessible and functional.

TEST_CASE("KernelEntry struct is constructible and accessible", "[unit][cudart][multi-kernel]") {
    KernelEntry entry;
    entry.name = "test_kernel";
    entry.arg_count = 3;
    entry.arg_byte_size = 24;
    REQUIRE(entry.name == "test_kernel");
    REQUIRE(entry.arg_count == 3);
    REQUIRE(entry.arg_byte_size == 24);
}

TEST_CASE("ManifestSection kernels vector supports push_back", "[unit][cudart][multi-kernel]") {
    ManifestSection ms;
    REQUIRE(ms.kernels.empty());
    KernelEntry e1;
    e1.name = "kernel_a";
    ms.kernels.push_back(e1);
    KernelEntry e2;
    e2.name = "kernel_b";
    ms.kernels.push_back(e2);
    REQUIRE(ms.kernels.size() == 2);
    REQUIRE(ms.kernels[0].name == "kernel_a");
    REQUIRE(ms.kernels[1].name == "kernel_b");
}

TEST_CASE("Multi-kernel: cuModuleGetFunction distinct-handle contract deferred to Phase 12.5",
          "[unit][cudart][multi-kernel]") {
    // Phase 12.4 scope = schema + backward-compat infrastructure.
    // Full cuModuleGetFunction multi-kernel name→handle mapping validation
    // requires v2 PTXIR writer + multi-entry fixture (Phase 12.5).
    SUCCEED("placeholder — full multi-kernel validation deferred to Phase 12.5");
}
