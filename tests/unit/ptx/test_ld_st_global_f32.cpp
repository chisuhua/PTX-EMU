// test_ld_st_global_f32.cpp
// =============================================================================
// Unit test (类型一): 全局内存 ld.global / st.global 行为验证
//
// 验证 ld.global.f32 和 st.global.f32 的基本正确性。
// 模拟 cute_rmsnorm 中 ld.global.nc.f32 和 st.global.f32 的路径。
//
// 注意：这需要 MemoryManager 基础设施，因此比 handler 直接测试更复杂。
// =============================================================================

#include "catch_amalgamated.hpp"

#include "memory/simple_memory.h"

#include <cstring>
#include <vector>

TEST_CASE("global memory: st.f32 then ld.f32 round-trip", "[global][memory][f32]") {
    SimpleMemory mem(4096);

    uint64_t addr = 0x100;
    float write_val = 42.5f;
    float read_val;

    mem.direct_access(addr, &write_val, sizeof(float), true);
    mem.direct_access(addr, &read_val, sizeof(float), false);

    REQUIRE(read_val == Catch::Approx(42.5f));
}

TEST_CASE("global memory: store/load multiple f32 values", "[global][memory][f32]") {
    SimpleMemory mem(4096);

    uint64_t base = 0x100;
    std::vector<float> write_vals = {1.0f, 2.0f, 3.5f, -4.2f};

    for (size_t i = 0; i < write_vals.size(); i++) {
        mem.direct_access(base + i * sizeof(float), &write_vals[i],
                          sizeof(float), true);
    }

    for (size_t i = 0; i < write_vals.size(); i++) {
        float read_val;
        mem.direct_access(base + i * sizeof(float), &read_val,
                          sizeof(float), false);
        REQUIRE(read_val == Catch::Approx(write_vals[i]));
    }
}

TEST_CASE("global memory: zero initialization", "[global][memory][f32]") {
    SimpleMemory mem(4096);

    uint64_t addr = 0x200;
    float read_val;
    mem.direct_access(addr, &read_val, sizeof(float), false);

    REQUIRE(read_val == Catch::Approx(0.0f));
}

TEST_CASE("global memory: large input (cute_rmsnorm scale)", "[global][memory][f32][cute]") {
    SimpleMemory mem(32768);

    uint64_t base = 0x1000;
    const int N = 768;

    for (int i = 0; i < N; i++) {
        float val = static_cast<float>((i % 100) - 50) * 0.1f;
        mem.direct_access(base + i * sizeof(float), &val, sizeof(float), true);
    }

    for (int i = 0; i < N; i++) {
        float expected = static_cast<float>((i % 100) - 50) * 0.1f;
        float read_val;
        mem.direct_access(base + i * sizeof(float), &read_val, sizeof(float), false);
        REQUIRE(read_val == Catch::Approx(expected));
    }
}

TEST_CASE("global memory: stride access (cute_rmsnorm thread pattern)", "[global][memory][f32][cute]") {
    SimpleMemory mem(32768);

    uint64_t base = 0x1000;
    const int blockSize = 256;
    const int N = 768;

    for (int i = 0; i < N; i++) {
        float val = static_cast<float>(i + 1);
        mem.direct_access(base + i * sizeof(float), &val, sizeof(float), true);
    }

    int tid = 0;
    float sum = 0.0f;
    for (int j = tid; j < N; j += blockSize) {
        float val;
        mem.direct_access(base + j * sizeof(float), &val, sizeof(float), false);
        sum += val * val;
    }

    REQUIRE(sum == Catch::Approx(329219.0f));
}

TEST_CASE("global memory: boundary access", "[global][memory][f32]") {
    SimpleMemory mem(4096);

    uint64_t last_addr = 4092;
    float val = 3.14f;
    mem.direct_access(last_addr, &val, sizeof(float), true);

    float read_val;
    mem.direct_access(last_addr, &read_val, sizeof(float), false);

    REQUIRE(read_val == Catch::Approx(3.14f));
}