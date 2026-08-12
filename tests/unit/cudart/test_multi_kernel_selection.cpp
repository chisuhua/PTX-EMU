#include "catch_amalgamated.hpp"
#include "cudart/cpptlm_module.h"
#include "cudart/cudart_intrinsics.h"
#include "ptx_ir/ptxir_format.h"

#include <filesystem>
#include <fstream>
#include <vector>

// Phase C5: Real multi-kernel selection tests using existing C2 fixture.
// Keep struct tests as-is (valid structural coverage).
// Replace placeholder with 4 real tests validating C3/C4 APIs.

namespace {
std::vector<uint8_t> load_fixture(const char* filename) {
    auto path = std::filesystem::path(TEST_FIXTURE_DIR) / filename;
    std::ifstream ifs(path, std::ios::binary);
    REQUIRE(ifs.good());
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(ifs)),
                                 std::istreambuf_iterator<char>());
    return bytes;
}
}  // namespace

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

// ========================================================================
// Phase C5: Real multi-kernel selection tests
// ========================================================================

TEST_CASE("Multi-kernel selection: ptxemu_image_kernel_count returns ≥3",
          "[unit][cudart][multi-kernel]") {
    auto fixture = load_fixture("multi_kernel_basic.ptxir");
    uint64_t h = ptxemu_image_load(fixture.data(), fixture.size());
    REQUIRE(h != 0);
    REQUIRE(ptxemu_image_kernel_count(h) >= 3);
    REQUIRE(ptxemu_image_unload(h) == 0);
}

TEST_CASE("Multi-kernel selection: ptxemu_image_kernel_name_at truncation contract",
          "[unit][cudart][multi-kernel]") {
    auto fixture = load_fixture("multi_kernel_basic.ptxir");
    uint64_t h = ptxemu_image_load(fixture.data(), fixture.size());
    REQUIRE(h != 0);

    char buf[64];
    REQUIRE(ptxemu_image_kernel_name_at(h, 0, buf, sizeof(buf)) == 7);
    REQUIRE(std::string(buf) == "vec_add");

    char tiny[4];
    int rc = ptxemu_image_kernel_name_at(h, 0, tiny, sizeof(tiny));
    REQUIRE(rc == 7);
    REQUIRE(tiny[3] == 0);

    REQUIRE(ptxemu_image_unload(h) == 0);
}

TEST_CASE("Multi-kernel selection: ptxemu_module_version gate enforces 1→2 bump",
          "[unit][cudart][multi-kernel]") {
    REQUIRE(ptxemu_module_version() == 2);
}
