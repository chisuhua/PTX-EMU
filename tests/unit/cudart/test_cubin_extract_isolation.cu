// test_cubin_extract_isolation.cu
// =============================================================================
// Unit test: concurrent PTX extraction isolation (fix-ptx-extraction-race)
//
// RED PHASE: this test MUST FAIL on unpatched code.
//
// Root cause (src/utils/cubin_utils.cpp:127-154): `extract_ptx_with_cuobjdump`
// extracts PTX content into the SHARED process current working directory and
// removes the extracted file after reading. Under concurrent calls (parallel
// ctest -j4), one call's `rm` can delete another call's in-flight file, and
// the fixed `__ptx_list_temp__` list file is overwritten/removed by other
// calls. The result is "Failed to open extracted PTX file" / empty PTX.
//
// This test starts N threads that ALL call `extract_ptx_with_cuobjdump` on
// the SAME CUDA test executable (which embeds PTX of the kernel below). Under
// the old shared-cwd implementation all threads race on identical temp file
// names; at least one call loses the race and returns empty/incomplete PTX.
// Under the fixed per-call `mkdtemp` implementation every call uses its own
// unique directory and always succeeds.
//
// Ref: openspec/changes/fix-ptx-extraction-race/
// =============================================================================

#include "catch_amalgamated.hpp"
#include "utils/cubin_utils.h"

#include <atomic>
#include <barrier>
#include <string>
#include <thread>
#include <vector>

#include <limits.h>
#include <unistd.h>

// A trivial kernel so the test executable embeds PTX (nvcc -keep --no-compress
// keeps the intermediate PTX in the fatbin). We never launch it; the fake
// libcudart.so is not exercised.
__global__ void kub_extract_isolation_kernel(float *out) {
    int i = threadIdx.x;
    out[i] = static_cast<float>(i);
}

namespace {

std::string self_exe_path() {
    char buf[PATH_MAX];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return "";
    }
    buf[n] = '\0';
    return std::string(buf);
}

bool ptx_contains_kernel(const std::string &ptx) {
    return ptx.find("kub_extract_isolation_kernel") != std::string::npos;
}

}  // namespace

// Core regression: N concurrent extractions of the same binary must each
// return complete, independent PTX. Fails on the shared-cwd implementation.
TEST_CASE("concurrent extraction returns complete PTX per call",
          "[unit][cudart][extract][race]") {
    const std::string exe = self_exe_path();
    REQUIRE_FALSE(exe.empty());
    REQUIRE(ptx_contains_kernel(extract_ptx_with_cuobjdump(exe)));

    constexpr int kNThreads = 8;
    constexpr int kRounds = 5;

    for (int round = 0; round < kRounds; ++round) {
        std::barrier gate(kNThreads);
        std::vector<std::string> results(kNThreads);
        std::atomic<bool> any_failed{false};
        std::vector<std::thread> threads;
        threads.reserve(kNThreads);

        for (int t = 0; t < kNThreads; ++t) {
            threads.emplace_back([&, t]() {
                gate.arrive_and_wait();
                results[t] = extract_ptx_with_cuobjdump(exe);
                if (!ptx_contains_kernel(results[t])) {
                    any_failed = true;
                }
            });
        }
        for (auto &th : threads) {
            th.join();
        }

        INFO("round #" << round << ": " << kNThreads
                       << " concurrent extractions of " << exe);
        REQUIRE_FALSE(any_failed.load());
        for (int t = 0; t < kNThreads; ++t) {
            REQUIRE_FALSE(results[t].empty());
            REQUIRE(ptx_contains_kernel(results[t]));
        }
    }
}