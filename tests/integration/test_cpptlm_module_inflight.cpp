#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
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

TEST_CASE("Concurrent launches (4 threads × 100 launches) serialize correctly", "[integration][cpptlm_module][inflight]") {
    auto p = fixture_path();
    REQUIRE(fs::exists(p));
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> bytes(sz);
    f.read(reinterpret_cast<char*>(bytes.data()), sz);

    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);

    auto worker = [handle]() -> int {
        void* args[] = {nullptr};
        for (int i = 0; i < 100; ++i) {
            int rc = ptxemu_image_execute(handle, 1, 1, 1, 32, 1, 1, 0, args, 0);
            if (rc != 0) return rc;
        }
        return 0;
    };

    std::vector<std::future<int>> futures;
    for (int t = 0; t < 4; ++t) {
        futures.push_back(std::async(std::launch::async, worker));
    }

    for (auto& fut : futures) {
        auto status = fut.wait_for(std::chrono::seconds(30));
        REQUIRE(status == std::future_status::ready);
        REQUIRE(fut.get() == 0);
    }

    REQUIRE(ptxemu_image_unload(handle) == 0);
}

TEST_CASE("In-flight unload returns non-zero, succeeds after kernel completes", "[integration][cpptlm_module][inflight]") {
    auto p = fixture_path();
    REQUIRE(fs::exists(p));
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> bytes(sz);
    f.read(reinterpret_cast<char*>(bytes.data()), sz);

    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);

    // Start kernel execution in background thread
    std::atomic<bool> kernel_started{false};
    std::atomic<int> kernel_rc{0};
    auto kernel_thread = std::thread([&, handle]() {
        void* args[] = {nullptr};
        kernel_started = true;
        kernel_rc = ptxemu_image_execute(handle, 1, 1, 1, 32, 1, 1, 0, args, 0);
    });

    // Wait for kernel to actually start
    while (!kernel_started) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Immediately try unload while kernel is in-flight — must return non-zero
    int unload_while_running = ptxemu_image_unload(handle);
    REQUIRE(unload_while_running != 0);

    // Wait for kernel to finish
    kernel_thread.join();
    REQUIRE(kernel_rc == 0);

    // After kernel completes, unload must succeed
    REQUIRE(ptxemu_image_unload(handle) == 0);
}
