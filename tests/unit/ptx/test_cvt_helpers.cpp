// test_cvt_helpers.cpp
// =============================================================================
// Unit test: 验证 CVT 4 个 helper 在抽离前后的行为一致性
// TDD 目的：抽离前先写测试，锁住当前行为；抽离后验证零行为变更
// =============================================================================

#include "catch_amalgamated.hpp"
#include "ptxsim/instructions/cvt/cvt_helpers.h"
#include <cmath>
#include <limits>

using ptxsim::cvt_helpers::float_to_half;
using ptxsim::cvt_helpers::half_to_float;
using ptxsim::cvt_helpers::round_half_to_even;
using ptxsim::cvt_helpers::should_saturate_uint32;

using Catch::Approx;

TEST_CASE("round_half_to_even basic cases", "[cvt][helpers][rounding]") {
    REQUIRE(round_half_to_even(0.0f) == 0.0f);
    REQUIRE(round_half_to_even(0.5f) == 0.0f); // banker's rounding
    REQUIRE(round_half_to_even(1.5f) == 2.0f);
    REQUIRE(round_half_to_even(2.5f) == 2.0f);
    REQUIRE(round_half_to_even(-0.5f) == -0.0f);
    REQUIRE(round_half_to_even(-1.5f) == -2.0f);
}

TEST_CASE("round_half_to_even edge cases", "[cvt][helpers][rounding]") {
    REQUIRE(std::isnan(
        round_half_to_even(std::numeric_limits<float>::quiet_NaN())));
    REQUIRE(round_half_to_even(std::numeric_limits<float>::infinity()) ==
            std::numeric_limits<float>::infinity());
}

TEST_CASE("half_to_float zero/inf/nan", "[cvt][helpers][half]") {
    REQUIRE(half_to_float(0x0000) == 0.0f);
    REQUIRE(half_to_float(0x8000) == -0.0f);
    REQUIRE(half_to_float(0x7C00) == std::numeric_limits<float>::infinity());
    REQUIRE(half_to_float(0xFC00) == -std::numeric_limits<float>::infinity());
    REQUIRE(std::isnan(half_to_float(0x7E00)));
}

TEST_CASE("half_to_float denormal smallest positive", "[cvt][helpers][half]") {
    // smallest positive denormal half = 2^-24
    float result = half_to_float(0x0001);
    REQUIRE(std::isfinite(result));
    REQUIRE(result > 0.0f);
    // 2^-24 = 5.9604644775390625e-08
    REQUIRE(result == Approx(5.9604644775390625e-08f).epsilon(0.01f));
}

TEST_CASE("half_to_float denormal largest", "[cvt][helpers][half]") {
    // largest denormal half = 0x03FF = (1 - 2^-10) * 2^-14
    float result = half_to_float(0x03FF);
    REQUIRE(std::isfinite(result));
    REQUIRE(result > 0.0f);
    REQUIRE(result == Approx(6.0975551605224609e-05f).epsilon(0.01f));
}

TEST_CASE("half_to_float negative denormal", "[cvt][helpers][half]") {
    // negative denormal = 0x8001
    float result = half_to_float(0x8001);
    REQUIRE(std::isfinite(result));
    REQUIRE(result < 0.0f);
    REQUIRE(result == Approx(-5.9604644775390625e-08f).epsilon(0.01f));
}

TEST_CASE("half_to_float denormal mid range monotonic",
          "[cvt][helpers][half]") {
    // Verify denormal path is monotonic and reasonably scaled
    float r1 = half_to_float(0x0001);
    float r2 = half_to_float(0x0002);
    float r3 = half_to_float(0x0100);
    float r4 = half_to_float(0x03FF);
    REQUIRE(r1 < r2);
    REQUIRE(r2 < r3);
    REQUIRE(r3 < r4);
    // r2 = 2*r1 (both denormal)
    REQUIRE(r2 == Approx(2.0f * r1).epsilon(0.01f));
}

TEST_CASE("float_to_half zero/inf/nan/denormal", "[cvt][helpers][half]") {
    REQUIRE(float_to_half(0.0f) == 0x0000);
    REQUIRE(float_to_half(-0.0f) == 0x8000);
    REQUIRE(float_to_half(std::numeric_limits<float>::infinity()) == 0x7C00);
    REQUIRE(float_to_half(-std::numeric_limits<float>::infinity()) == 0xFC00);
    REQUIRE(float_to_half(std::numeric_limits<float>::quiet_NaN()) == 0x7E00);
}

TEST_CASE("should_saturate_uint32 boundaries", "[cvt][helpers][sat]") {
    REQUIRE_FALSE(should_saturate_uint32(0.0f, 4294967295.0f));
    REQUIRE_FALSE(should_saturate_uint32(100.5f, 4294967295.0f));
    // Pre-existing inline bug: strict `<` for sat_high + float32 rounds
    // 4294967295.0f up to 4294967296.0f. With sat_high == 4294967295.0f,
    // the upper-bound check always fails; the function never returns true
    // for these inputs. A future fix is tracked separately.
    REQUIRE_FALSE(should_saturate_uint32(4294967295.0f, 4294967295.0f));
    REQUIRE_FALSE(should_saturate_uint32(1e10f, 4294967295.0f));
    REQUIRE_FALSE(should_saturate_uint32(std::numeric_limits<float>::infinity(),
                                         4294967295.0f));
    REQUIRE(should_saturate_uint32(5e9f, 1e10f));
    REQUIRE_FALSE(should_saturate_uint32(5e9f, 5e9f));
    REQUIRE_FALSE(should_saturate_uint32(5e9f, 4e9f));
}