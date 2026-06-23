// test_half_utils.cpp
// =============================================================================
// Unit test: 验证 half_utils.h f16 ↔ f32 转换的 denormal 路径正确性
// 修复 half-precision-bugfix 后，half_utils 与 cvt_helpers 应行为一致
//
// IEEE 754 half precision:
//   - 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits
//   - Denormal: exp=0, value = mantissa × 2^-24 (mantissa 1..1023)
//   - Smallest denormal = 2^-24 ≈ 5.96e-8
//   - Largest denormal = (1023/1024) × 2^-14 ≈ 6.097e-5
// =============================================================================

#include "catch_amalgamated.hpp"
#include "ptxsim/utils/half_utils.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

// f16_to_f32 / f32_to_f16 are declared in global namespace per
// include/ptxsim/utils/half_utils.h (API preserved per proposal §Out of Scope).

// ----------------------------------------------------------------------------
// f16_to_f32 — special values
// ----------------------------------------------------------------------------
TEST_CASE("f16_to_f32 zero/inf/nan", "[half][utils]") {
    REQUIRE(f16_to_f32(0x0000) == 0.0f);
    REQUIRE(f16_to_f32(0x8000) == -0.0f);
    REQUIRE(f16_to_f32(0x7C00) == std::numeric_limits<float>::infinity());
    REQUIRE(f16_to_f32(0xFC00) == -std::numeric_limits<float>::infinity());
    REQUIRE(std::isnan(f16_to_f32(0x7E00)));
}

// ----------------------------------------------------------------------------
// f16_to_f32 — denormal path (RED: this is the buggy path)
// ----------------------------------------------------------------------------
TEST_CASE("f16_to_f32 denormal smallest positive", "[half][utils][denormal]") {
    // smallest positive denormal half = 2^-24 ≈ 5.96e-8
    float result = f16_to_f32(0x0001);
    REQUIRE(std::isfinite(result));
    REQUIRE(result > 0.0f);
    REQUIRE(result == Catch::Approx(5.9604644775390625e-08f).epsilon(0.01f));
}

TEST_CASE("f16_to_f32 denormal largest", "[half][utils][denormal]") {
    // largest denormal half = 0x03FF ≈ 6.097e-5
    float result = f16_to_f32(0x03FF);
    REQUIRE(std::isfinite(result));
    REQUIRE(result > 0.0f);
    REQUIRE(result == Catch::Approx(6.0975551605224609e-05f).epsilon(0.01f));
}

TEST_CASE("f16_to_f32 denormal negative", "[half][utils][denormal]") {
    // negative denormal = 0x8001
    float result = f16_to_f32(0x8001);
    REQUIRE(std::isfinite(result));
    REQUIRE(result < 0.0f);
    REQUIRE(result == Catch::Approx(-5.9604644775390625e-08f).epsilon(0.01f));
}

TEST_CASE("f16_to_f32 denormal mid-range", "[half][utils][denormal]") {
    // 0x0100 (mantissa = 256): value = 256 × 2^-24 = 2^-16
    float result = f16_to_f32(0x0100);
    REQUIRE(std::isfinite(result));
    REQUIRE(result == Catch::Approx(1.52587890625e-05f).epsilon(0.01f));
}

// ----------------------------------------------------------------------------
// f16_to_f32 — normal boundary
// ----------------------------------------------------------------------------
TEST_CASE("f16_to_f32 normal boundary", "[half][utils]") {
    // 0x3C00 = 1.0 in half
    REQUIRE(f16_to_f32(0x3C00) == 1.0f);
    // 0x4000 = 2.0
    REQUIRE(f16_to_f32(0x4000) == 2.0f);
    // 0x7BFF = largest normal ≈ 65504
    REQUIRE(f16_to_f32(0x7BFF) == Catch::Approx(65504.0f).epsilon(0.001f));
    // 0x0001 (smallest denormal, exact IEEE value 2^-24)
    // also verified by the denormal test above
}

// ----------------------------------------------------------------------------
// f32_to_f16 — special values
// ----------------------------------------------------------------------------
TEST_CASE("f32_to_f16 zero/inf/nan", "[half][utils]") {
    REQUIRE(f32_to_f16(0.0f) == 0x0000);
    REQUIRE(f32_to_f16(-0.0f) == 0x8000);
    REQUIRE(f32_to_f16(std::numeric_limits<float>::infinity()) == 0x7C00);
    REQUIRE(f32_to_f16(-std::numeric_limits<float>::infinity()) == 0xFC00);
    REQUIRE(f32_to_f16(std::numeric_limits<float>::quiet_NaN()) == 0x7E00);
}

// ----------------------------------------------------------------------------
// f32_to_f16 — normal boundary
// ----------------------------------------------------------------------------
TEST_CASE("f32_to_f16 normal boundary", "[half][utils]") {
    REQUIRE(f32_to_f16(1.0f) == 0x3C00);
    REQUIRE(f32_to_f16(2.0f) == 0x4000);
    REQUIRE(f32_to_f16(65504.0f) == 0x7BFF);
    REQUIRE(f32_to_f16(65536.0f) == 0x7C00);  // overflows to +Inf
    REQUIRE(f32_to_f16(-65536.0f) == 0xFC00); // overflows to -Inf
}

// ----------------------------------------------------------------------------
// f32_to_f16 — float subnormal input
//
// IEEE 754 float subnormals have exp=0, mantissa != 0.  Every float
// subnormal has value < 2^-126, which is far below the smallest half
// subnormal (2^-24).  So per IEEE 754 round-to-nearest-even, all float
// subnormals must round to signed zero in half precision.  The previous
// half_utils implementation dropped the sign, and cvt_helpers
// incorrectly renormalized the mantissa producing a huge positive value.
// ----------------------------------------------------------------------------
TEST_CASE("f32_to_f16 float subnormal smallest positive",
          "[half][utils][denormal]") {
    // 1.4e-45 (smallest float subnormal): rounds to +0
    REQUIRE(f32_to_f16(1.4e-45f) == 0x0000);
}

TEST_CASE("f32_to_f16 float subnormal mid-range", "[half][utils][denormal]") {
    // 1.0e-40: < 2^-126 << 2^-24, rounds to +0
    REQUIRE(f32_to_f16(1.0e-40f) == 0x0000);
}

TEST_CASE("f32_to_f16 float subnormal negative", "[half][utils][denormal]") {
    // -1.0e-40: rounds to -0 (preserves sign)
    REQUIRE(f32_to_f16(-1.0e-40f) == 0x8000);
}

// ----------------------------------------------------------------------------
// f32_to_f16 — normal float underflowing to half subnormal
//
// Normal float values in the range [2^-25, 2^-14) are representable as
// half subnormals (exp=0, non-zero mantissa).
// ----------------------------------------------------------------------------
TEST_CASE("f32_to_f16 normal-underflow becomes half subnormal",
          "[half][utils][denormal]") {
    // 2^-20 = 9.5367e-7: float normal, half subnormal
    uint16_t result = f32_to_f16(9.5367431640625e-07f);
    REQUIRE((result & 0x7C00) == 0x0000); // exp = 0 (denormal)
    REQUIRE((result & 0x8000) == 0x0000); // sign = 0
    REQUIRE((result & 0x3FF) != 0x0000);  // mantissa != 0
}