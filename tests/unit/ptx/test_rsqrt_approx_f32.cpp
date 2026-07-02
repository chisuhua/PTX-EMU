// test_rsqrt_approx_f32.cpp
// =============================================================================
// Unit test (类型一): rsqrt.approx.f32 数学正确性验证
//
// 验证 cute_rmsnorm 中使用的 rsqrt.approx.f32 等价数学操作。
// RsqrtHandler 实现 (math.cpp:129):
//   1.0 / std::sqrt(x)  ← NOT NVIDIA approximate rsqrt
// =============================================================================

#include "catch_amalgamated.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

static float sim_rsqrt_f32(float x) {
    return 1.0f / std::sqrt(x);
}

TEST_CASE("rsqrt.approx.f32: basic values", "[rsqrt][f32][math]") {
    REQUIRE(sim_rsqrt_f32(1.0f) == Catch::Approx(1.0f));
    REQUIRE(sim_rsqrt_f32(4.0f) == Catch::Approx(0.5f));
    REQUIRE(sim_rsqrt_f32(100.0f) == Catch::Approx(0.1f));
    REQUIRE(sim_rsqrt_f32(0.25f) == Catch::Approx(2.0f));
}

TEST_CASE("rsqrt.approx.f32: cute_rmsnorm epsilon path", "[rsqrt][f32][math][cute]") {
    // mean_sq ≈ 1.0 → rsqrt ≈ 1.0
    REQUIRE(sim_rsqrt_f32(1.0f) == Catch::Approx(1.0f));
    // mean_sq = eps = 1e-6 → rsqrt ≈ 1000
    REQUIRE(sim_rsqrt_f32(1e-6f) == Catch::Approx(1000.0f));
}

TEST_CASE("rsqrt.approx.f32: edge cases", "[rsqrt][f32][math]") {
    REQUIRE(std::isinf(sim_rsqrt_f32(0.0f)));
    REQUIRE(std::isnan(sim_rsqrt_f32(-1.0f)));
    REQUIRE(sim_rsqrt_f32(1e20f) < 1e-9f);
}

TEST_CASE("rsqrt.approx.f32: precision vs double reference", "[rsqrt][f32][math]") {
    for (float x = 0.1f; x <= 10.0f; x += 0.5f) {
        float result = sim_rsqrt_f32(x);
        double ref = 1.0 / std::sqrt(static_cast<double>(x));
        float rel_err = std::abs(result - static_cast<float>(ref))
                       / static_cast<float>(ref);
        REQUIRE(rel_err < 1e-6f);
    }
}