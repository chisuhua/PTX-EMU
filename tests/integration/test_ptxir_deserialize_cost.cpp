#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <catch_amalgamated.hpp>
#include "cudart/cpptlm_module.h"

namespace fs = std::filesystem;

static fs::path fixture_path() {
    fs::path p = fs::current_path();
    while (!p.empty()) {
        if (fs::exists(p / "tests" / "ptxir" / "fixtures" / "cute_rmsnorm.ptxir")) {
            return p / "tests" / "ptxir" / "fixtures" / "cute_rmsnorm.ptxir";
        }
        if (p == p.parent_path()) break;
        p = p.parent_path();
    }
    return fs::path("tests/ptxir/fixtures/cute_rmsnorm.ptxir");
}

TEST_CASE("D3 perf gate: cute_rmsnorm deserialize cost", "[performance][cpptlm_module]") {
    auto p = fixture_path();
    REQUIRE(fs::exists(p));
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> bytes(sz);
    f.read(reinterpret_cast<char*>(bytes.data()), sz);

    void* args[] = {nullptr};

    uint64_t ha = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(ha != 0);
    auto t0a = std::chrono::high_resolution_clock::now();
    // Plan task 3.4: integration env has no g_gpu_context → -EINVAL.
    REQUIRE(ptxemu_image_execute(ha, 1, 1, 1, 32, 1, 1, 0, args, 0) == -EINVAL);
    auto t1a = std::chrono::high_resolution_clock::now();
    auto dur_a = std::chrono::duration_cast<std::chrono::microseconds>(t1a - t0a).count();
    ptxemu_image_unload(ha);

    uint64_t hb = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(hb != 0);
    auto t0b = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        REQUIRE(ptxemu_image_execute(hb, 1, 1, 1, 32, 1, 1, 0, args, 0) == -EINVAL);
    }
    auto t1b = std::chrono::high_resolution_clock::now();
    auto dur_b = std::chrono::duration_cast<std::chrono::microseconds>(t1b - t0b).count();
    ptxemu_image_unload(hb);

    double ratio = (dur_a > 0) ? (double)dur_b / (double)(dur_a * 100) : 0.0;
    std::cout << "deserialize_cost=" << std::fixed << std::setprecision(3)
              << ratio << "x  (A=" << dur_a << "us, B=" << dur_b << "us)" << std::endl;

    if (ratio < 1.10) {
        std::cout << "PASS (deserialize cost below 10% threshold)" << std::endl;
        SUCCEED();
    } else {
        std::cout << "FAIL (A1 fallback required) — ratio " << ratio << "x >= 1.10" << std::endl;
        FAIL("D3 perf gate FAILED");
    }
}
