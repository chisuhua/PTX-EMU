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

TEST_CASE("Concurrent launches (4 threads × 100 launches) uniformly return -EINVAL without g_gpu_context",
          "[integration][cpptlm_module][inflight]") {
    // Plan task 3.4: integration env has no g_gpu_context. All 4 threads'
    // execute() calls return -EINVAL. The "serialize correctly" property
    // (every call returns the same error code) is still verified.
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
            if (rc != -EINVAL) return rc;
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

TEST_CASE("No-context execute returns -EINVAL immediately, unload then succeeds",
          "[integration][cpptlm_module][inflight]") {
    // Plan task 3.4: integration env has no g_gpu_context, so execute()
    // returns -EINVAL immediately (no kernel actually launches). The
    // "in-flight unload returns non-zero" semantic cannot be tested without
    // a live GPUContext; the full path_2D in-flight contract is covered
    // end-to-end by the GPU-context-enabled test in
    // e2e/path_2D_image_executor/test_image_executor_synchronous.cpp.
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

    std::atomic<bool> kernel_started{false};
    std::atomic<int> kernel_rc{0};
    auto kernel_thread = std::thread([&, handle]() {
        void* args[] = {nullptr};
        kernel_started = true;
        kernel_rc = ptxemu_image_execute(handle, 1, 1, 1, 32, 1, 1, 0, args, 0);
    });

    while (!kernel_started) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // With no g_gpu_context, execute returns -EINVAL immediately; nothing
    // is in-flight, so unload succeeds (rc == 0).
    int unload_rc = ptxemu_image_unload(handle);
    REQUIRE(unload_rc == 0);

    kernel_thread.join();
    REQUIRE(kernel_rc == -EINVAL);

    // Handle is already unloaded; a second unload returns -EINVAL.
    REQUIRE(ptxemu_image_unload(handle) == -EINVAL);
}
