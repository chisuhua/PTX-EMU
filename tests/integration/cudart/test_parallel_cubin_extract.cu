// test_parallel_cubin_extract.cu
// =============================================================================
// Integration test: repeated concurrent real-binary PTX extraction
// (fix-ptx-extraction-race)
//
// RED PHASE: this test MUST FAIL on unpatched code.
//
// This is the integration-level stress case for the helper used by the CUDA
// runtime interception path. The executable contains a real CUDA kernel, and
// multiple host threads repeatedly invoke the same extraction path at once.
// Every result must contain a complete PTX entry for the embedded kernel.
//
// The six pre-existing ctest failures are process-level manifestations of the
// same shared-workspace race. This test provides a focused, repeatable guard
// for the helper without recursively invoking ctest from inside a test.
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

__global__ void kub_parallel_extract_kernel(int *out) {
    const int i = threadIdx.x;
    out[i] = i * 2;
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

bool has_expected_kernel(const std::string &ptx) {
    return ptx.find("kub_parallel_extract_kernel") != std::string::npos;
}

}  // namespace

TEST_CASE("parallel extraction remains isolated across repeated rounds",
          "[integration][cudart][extract][race]") {
    const std::string exe = self_exe_path();
    REQUIRE_FALSE(exe.empty());
    REQUIRE(has_expected_kernel(extract_ptx_with_cuobjdump(exe)));

    constexpr int kWorkers = 12;
    constexpr int kRounds = 8;
    std::atomic<int> completed{0};

    for (int round = 0; round < kRounds; ++round) {
        std::barrier gate(kWorkers);
        std::vector<std::string> results(kWorkers);
        std::vector<std::thread> workers;
        workers.reserve(kWorkers);

        for (int worker = 0; worker < kWorkers; ++worker) {
            workers.emplace_back([&, worker]() {
                gate.arrive_and_wait();
                results[worker] = extract_ptx_with_cuobjdump(exe);
                if (has_expected_kernel(results[worker])) {
                    completed.fetch_add(1);
                }
            });
        }
        for (auto &worker : workers) {
            worker.join();
        }

        INFO("round #" << round << ": " << kWorkers
                       << " concurrent extractions");
        for (const auto &ptx : results) {
            REQUIRE(has_expected_kernel(ptx));
        }
    }

    REQUIRE(completed.load() == kWorkers * kRounds);
}