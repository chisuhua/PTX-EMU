// test_fixture_load.cpp
// Phase C2: multi-entry fixture load tests
#include "catch_amalgamated.hpp"
#include "ptx_ir/ptxir_reader.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

TEST_CASE("Multi-entry fixture has ≥3 kernels", "[unit][ptxir][fixture]") {
    auto path = std::filesystem::path(TEST_FIXTURE_DIR) / "multi_kernel_basic.ptxir";
    REQUIRE(std::filesystem::exists(path));

    std::ifstream ifs(path, std::ios::binary);
    REQUIRE(ifs.good());

    PtxirReader reader(ifs);
    reader.read();  // parse the full PTXIR
    const auto& manifest = reader.get_manifest();

    REQUIRE(manifest.kernels.size() >= 3);
    REQUIRE(manifest.kernels[0].name == "vec_add");
    REQUIRE(manifest.kernels[1].name == "mat_mul");
    REQUIRE(manifest.kernels[2].name == "reduce_sum");
}

TEST_CASE("Multi-entry fixture round-trip is stable", "[unit][ptxir][fixture]") {
    auto path = std::filesystem::path(TEST_FIXTURE_DIR) / "multi_kernel_basic.ptxir";
    std::ifstream ifs(path, std::ios::binary);
    REQUIRE(ifs.good());

    PtxirReader reader1(ifs);
    reader1.read();
    const auto m1 = reader1.get_manifest();

    // Re-open to re-read
    std::ifstream ifs2(path, std::ios::binary);
    PtxirReader reader2(ifs2);
    reader2.read();
    const auto m2 = reader2.get_manifest();

    REQUIRE(m1.kernels.size() == m2.kernels.size());
    for (size_t i = 0; i < m1.kernels.size(); ++i) {
        REQUIRE(m1.kernels[i].name == m2.kernels[i].name);
        REQUIRE(m1.kernels[i].arg_count == m2.kernels[i].arg_count);
    }
}
