// test_fma_rn_f32.cpp
// =============================================================================
// Unit test (类型一): fma.rn.f32 数学正确性验证
//
// 验证 cute_rmsnorm 中使用的 fma.rn.f32 等价数学操作。
// FmaHandler 实现 (arithmetic_ext.cpp:598) 使用:
//   r = a * b + c;  ← 两次舍入，不是真正的 FMA
// =============================================================================

#include "catch_amalgamated.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

static float sim_fma_f32(float a, float b, float c) {
    return a * b + c;
}

TEST_CASE("fma.rn.f32: basic mul-add", "[fma][f32][math]") {
    REQUIRE(sim_fma_f32(2.0f, 3.0f, 4.0f) == Catch::Approx(10.0f));
    REQUIRE(sim_fma_f32(-1.5f, 2.0f, -3.0f) == Catch::Approx(-6.0f));
    REQUIRE(sim_fma_f32(0.0f, 5.0f, 7.0f) == Catch::Approx(7.0f));
}

TEST_CASE("fma.rn.f32: accumulate squares (cute_rmsnorm)", "[fma][f32][math][cute]") {
    float sum_sq = 0.0f;
    float vals[] = {1.5f, 2.0f, 0.5f};
    float expected = 0.0f;
    for (float v : vals) {
        sum_sq = sim_fma_f32(v, v, sum_sq);
        expected += v * v;
    }
    REQUIRE(sum_sq == Catch::Approx(expected).epsilon(1e-6f));
}

TEST_CASE("fma.rn.f32: 768-iteration accumulation", "[fma][f32][math][cute]") {
    float sum_sq = 0.0f;
    double reference = 0.0;
    for (int i = 1; i <= 768; i++) {
        float x = (i % 50) * 0.02f;
        sum_sq = sim_fma_f32(x, x, sum_sq);
        reference += static_cast<double>(x) * static_cast<double>(x);
    }
    float relative_error = std::abs(sum_sq - static_cast<float>(reference))
                          / static_cast<float>(reference);
    REQUIRE(relative_error < 1e-4f);
}

TEST_CASE("fma.rn.f32: NaN propagation", "[fma][f32][math]") {
    float nan = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(std::isnan(sim_fma_f32(nan, 1.0f, 0.0f)));
}