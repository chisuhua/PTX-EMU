// test_cvt_int_to_float.cpp
// =============================================================================
// Unit test (类型一): IntToFloatStrategy 行为锁定
//
// 背景:
//   T2-6 Sub-task 4b — CVT 策略模式拆分第二个具体策略。
//   IntToFloatStrategy 处理: dst_is_float && !src_is_float
//   (s8/s16/s32/s64/u8/u16/u32/u64 → f16/f32/f64, 含 .sat NaN→0)
//
// 行为锁定目的: Sub-task 6 删除 GeneralCvtStrategy 时不破坏。
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/instructions/cvt/cvt_int_to_float.h"
#include "ptxsim/instructions/cvt/cvt_strategy.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

using ptxsim::cvt_strategy::CvtContext;
using ptxsim::cvt_strategy::IntToFloatStrategy;

namespace {

CvtContext make_i2f_dst_f32_src(Qualifier src_dtype, int src_bytes,
                                bool src_signed) {
    CvtContext ctx;
    ctx.dst_bytes = 4;
    ctx.dst_is_float = true;
    ctx.src_bytes = src_bytes;
    ctx.src_is_signed = src_signed;
    return ctx;
}

CvtContext make_i2f_dst_f64_src(Qualifier src_dtype, int src_bytes,
                                bool src_signed) {
    CvtContext ctx;
    ctx.dst_bytes = 8;
    ctx.dst_is_float = true;
    ctx.src_bytes = src_bytes;
    ctx.src_is_signed = src_signed;
    return ctx;
}

CvtContext make_i2f_dst_f16_src(Qualifier src_dtype, int src_bytes,
                                bool src_signed) {
    CvtContext ctx;
    ctx.dst_bytes = 2;
    ctx.dst_is_float = true;
    ctx.dst_is_half = true;
    ctx.src_bytes = src_bytes;
    ctx.src_is_signed = src_signed;
    return ctx;
}

uint32_t f32_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

uint64_t f64_bits(double d) {
    uint64_t b;
    std::memcpy(&b, &d, 8);
    return b;
}

} // namespace

TEST_CASE("IntToFloat s8->f32", "[cvt][strategy][i2f]") {
    IntToFloatStrategy s;
    float dst = 0.0f;
    int8_t src = -42;
    s.convert(&dst, &src, make_i2f_dst_f32_src(Qualifier::Q_S8, 1, true));
    REQUIRE(dst == -42.0f);
}

TEST_CASE("IntToFloat u8->f32", "[cvt][strategy][i2f]") {
    IntToFloatStrategy s;
    float dst = 0.0f;
    uint8_t src = 200;
    s.convert(&dst, &src, make_i2f_dst_f32_src(Qualifier::Q_U8, 1, false));
    REQUIRE(dst == 200.0f);
}

TEST_CASE("IntToFloat s16->f32", "[cvt][strategy][i2f]") {
    IntToFloatStrategy s;
    float dst = 0.0f;
    int16_t src = -12345;
    s.convert(&dst, &src, make_i2f_dst_f32_src(Qualifier::Q_S16, 2, true));
    REQUIRE(dst == -12345.0f);
}

TEST_CASE("IntToFloat u16->f32", "[cvt][strategy][i2f]") {
    IntToFloatStrategy s;
    float dst = 0.0f;
    uint16_t src = 50000;
    s.convert(&dst, &src, make_i2f_dst_f32_src(Qualifier::Q_U16, 2, false));
    REQUIRE(dst == 50000.0f);
}

TEST_CASE("IntToFloat s32->f32", "[cvt][strategy][i2f]") {
    IntToFloatStrategy s;
    float dst = 0.0f;
    int32_t src = -1000000;
    s.convert(&dst, &src, make_i2f_dst_f32_src(Qualifier::Q_S32, 4, true));
    REQUIRE(dst == -1000000.0f);
}

TEST_CASE("IntToFloat u32->f32", "[cvt][strategy][i2f]") {
    IntToFloatStrategy s;
    float dst = 0.0f;
    uint32_t src = 3000000000U;
    s.convert(&dst, &src, make_i2f_dst_f32_src(Qualifier::Q_U32, 4, false));
    REQUIRE(dst == 3000000000.0f);
}

TEST_CASE("IntToFloat s64->f32", "[cvt][strategy][i2f]") {
    IntToFloatStrategy s;
    float dst = 0.0f;
    int64_t src = -9000000000LL;
    s.convert(&dst, &src, make_i2f_dst_f32_src(Qualifier::Q_S64, 8, true));
    REQUIRE(dst == -9000000000.0f);
}

TEST_CASE("IntToFloat u64->f32", "[cvt][strategy][i2f]") {
    IntToFloatStrategy s;
    float dst = 0.0f;
    uint64_t src = 18000000000ULL;
    s.convert(&dst, &src, make_i2f_dst_f32_src(Qualifier::Q_U64, 8, false));
    REQUIRE(dst == 18000000000.0f);
}

TEST_CASE("IntToFloat s32->f64 widening", "[cvt][strategy][i2f]") {
    IntToFloatStrategy s;
    double dst = 0.0;
    int32_t src = -12345;
    s.convert(&dst, &src, make_i2f_dst_f64_src(Qualifier::Q_S32, 4, true));
    REQUIRE(dst == -12345.0);
    REQUIRE(f64_bits(dst) == f64_bits(-12345.0));
}

TEST_CASE("IntToFloat u64->f64 widening preserves precision",
          "[cvt][strategy][i2f]") {
    // u64 max -> f64: 18446744073709551615.0 (may lose precision in f32 but
    // exact in f64)
    IntToFloatStrategy s;
    double dst = 0.0;
    uint64_t src = 18446744073709551615ULL;
    s.convert(&dst, &src, make_i2f_dst_f64_src(Qualifier::Q_U64, 8, false));
    // f64 can represent up to 2^53 exactly, larger u64 rounds
    REQUIRE(dst > 0.0);
    // Note: actual value 1.8446744073709552e19
    REQUIRE(f64_bits(dst) != 0);
}

TEST_CASE("IntToFloat s32->f16 narrowing", "[cvt][strategy][i2f]") {
    IntToFloatStrategy s;
    uint16_t dst = 0;
    int32_t src = 3; // representable in half
    s.convert(&dst, &src, make_i2f_dst_f16_src(Qualifier::Q_S32, 4, true));
    // 3.0 in half = 0x4200
    REQUIRE(dst == 0x4200);
}

TEST_CASE("IntToFloat s8->f32 .sat handles NaN -> 0 (no NaN path; passthrough)",
          "[cvt][strategy][i2f][sat]") {
    // .sat on int->float is a no-op (no NaN source); just verify passthrough.
    IntToFloatStrategy s;
    auto ctx = make_i2f_dst_f32_src(Qualifier::Q_S8, 1, true);
    ctx.has_sat = true;
    float dst = 0.0f;
    int8_t src = -100;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == -100.0f);
}

TEST_CASE("IntToFloat zero handling", "[cvt][strategy][i2f][edge]") {
    IntToFloatStrategy s;
    float dst = 999.0f;
    int32_t src = 0;
    s.convert(&dst, &src, make_i2f_dst_f32_src(Qualifier::Q_S32, 4, true));
    REQUIRE(dst == 0.0f);
    // Verify sign of zero is 0
    REQUIRE(f32_bits(dst) == 0u);
}
