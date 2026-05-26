/**
 * @file test_multiwarp_barrier_divergence.cpp
 * @brief Mode 3C: E2E test for multi-warp CTA barrier divergence
 * @date 2026-05-26
 *
 * Tests that a 64-thread CTA (2 warps) with warp-level divergence before
 * __syncthreads() correctly forces reconvergence at the barrier.
 * Per sm90_100.md:294: "bar.sync — 未汇合的 Warp 会在此被强制汇合"
 *
 * This test runs the standalone binary and verifies correct behavior.
 */

#include "catch_amalgamated.hpp"
#include <cstdio>
#include <string>

#ifndef TEST_BINARY
#define TEST_BINARY "build/bin/test_divergence_sync_standalone"
#endif

TEST_CASE("Multi-warp barrier divergence: warp-level divergence reconverges at barrier",
          "[e2e][barrier][divergence][multi-warp]")
{
    // Run standalone binary - multi-warp barrier divergence test
    // Uses LD_LIBRARY_PATH to load fake libcudart.so
    std::string cmd = "PTX_LOG_LEVEL=error LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH "
                      "timeout 30 "
                      TEST_BINARY " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    REQUIRE(pipe != nullptr);

    std::string output;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        output += buf;
    }
    int ret = pclose(pipe);

    INFO("Standalone binary output (" << output.size() << " bytes):");
    INFO(output);

    bool has_pass = output.find("=== Result: PASS ===") != std::string::npos;
    bool has_fail = output.find("=== Result: FAIL ===") != std::string::npos;

    INFO("Has PASS: " << has_pass << ", Has FAIL: " << has_fail);

    CHECK(has_pass);
}