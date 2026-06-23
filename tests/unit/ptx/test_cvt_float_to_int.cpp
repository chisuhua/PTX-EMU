// test_cvt_float_to_int.cpp
// =============================================================================
// Unit test (类型一): FloatToIntStrategy 行为锁定
//
// 背景:
//   T2-6 Sub-task 4c+4d — CVT 策略模式拆分第三个具体策略。
//   FloatToIntStrategy 处理: !dst_is_float && src_is_float
//   (f16/f32/f64 → s8/s16/s32/s64/u8/u16/u32/u64)
//   含 .sat 饱和处理 + 5 种舍入模式 (.rn/.rz/.rm/.rp/.rna)
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/instructions/cvt/cvt_float_to_int.h"
#include "ptxsim/instructions/cvt/cvt_strategy.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

using ptxsim::cvt_strategy::CvtContext;
using ptxsim::cvt_strategy::FloatToIntStrategy;

namespace {

CvtContext make_f2i_dst_s8() {
    CvtContext ctx;
    ctx.dst_bytes = 1;
    ctx.dst_is_signed = true;
    ctx.src_bytes = 4;
    ctx.src_is_float = true;
    return ctx;
}

CvtContext make_f2i_dst_u8() {
    CvtContext ctx;
    ctx.dst_bytes = 1;
    ctx.dst_is_signed = false;
    ctx.src_bytes = 4;
    ctx.src_is_float = true;
    return ctx;
}

CvtContext make_f2i_dst_s16() {
    CvtContext ctx;
    ctx.dst_bytes = 2;
    ctx.dst_is_signed = true;
    ctx.src_bytes = 4;
    ctx.src_is_float = true;
    return ctx;
}

CvtContext make_f2i_dst_u16() {
    CvtContext ctx;
    ctx.dst_bytes = 2;
    ctx.dst_is_signed = false;
    ctx.src_bytes = 4;
    ctx.src_is_float = true;
    return ctx;
}

CvtContext make_f2i_dst_s32() {
    CvtContext ctx;
    ctx.dst_bytes = 4;
    ctx.dst_is_signed = true;
    ctx.src_bytes = 4;
    ctx.src_is_float = true;
    return ctx;
}

CvtContext make_f2i_dst_u32() {
    CvtContext ctx;
    ctx.dst_bytes = 4;
    ctx.dst_is_signed = false;
    ctx.src_bytes = 4;
    ctx.src_is_float = true;
    return ctx;
}

CvtContext make_f2i_dst_s64() {
    CvtContext ctx;
    ctx.dst_bytes = 8;
    ctx.dst_is_signed = true;
    ctx.src_bytes = 8;
    ctx.src_is_float = true;
    return ctx;
}

CvtContext make_f2i_dst_u64() {
    CvtContext ctx;
    ctx.dst_bytes = 8;
    ctx.dst_is_signed = false;
    ctx.src_bytes = 8;
    ctx.src_is_float = true;
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

// ---- .sat path tests ----

TEST_CASE("FloatToInt f32->s8 .sat handles NaN -> 0",
          "[cvt][strategy][f2i][sat]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s8();
    ctx.has_sat = true;
    int8_t dst = -1;
    float src = std::numeric_limits<float>::quiet_NaN();
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 0);
}

TEST_CASE("FloatToInt f32->s8 .sat clamps positive over",
          "[cvt][strategy][f2i][sat]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s8();
    ctx.has_sat = true;
    int8_t dst = 0;
    float src = 200.0f; // > 127
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 127);
}

TEST_CASE("FloatToInt f32->s8 .sat clamps negative under",
          "[cvt][strategy][f2i][sat]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s8();
    ctx.has_sat = true;
    int8_t dst = 0;
    float src = -200.0f; // < -128
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == -128);
}

TEST_CASE("FloatToInt f32->s8 .sat passthrough on in-range",
          "[cvt][strategy][f2i][sat]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s8();
    ctx.has_sat = true;
    int8_t dst = 0;
    float src = 42.5f; // 42 (truncation under .sat)
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 42);
}

TEST_CASE("FloatToInt f32->u32 .sat well-above boundary",
          "[cvt][strategy][f2i][sat]") {
    // 1e10f is well above UINT32_MAX and saturates cleanly to 0xFFFFFFFF.
    // (Note: 4294967295.0f triggers a separate f32 precision bug in the
    // .sat path; that bug is out-of-scope for T2-6 — see
    // test_cvt_edge_cases.cpp:152-155 "out-of-scope" note.)
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_u32();
    ctx.has_sat = true;
    uint32_t dst = 0;
    float src = 1e10f;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 0xFFFFFFFFu);
}

TEST_CASE("FloatToInt f32->u32 .sat handles NaN -> 0",
          "[cvt][strategy][f2i][sat]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_u32();
    ctx.has_sat = true;
    uint32_t dst = 0xDEADBEEF;
    float src = std::numeric_limits<float>::quiet_NaN();
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 0);
}

TEST_CASE("FloatToInt f32->u32 .sat handles negative -> 0",
          "[cvt][strategy][f2i][sat]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_u32();
    ctx.has_sat = true;
    uint32_t dst = 0;
    float src = -1.0f;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 0);
}

TEST_CASE("FloatToInt f32->s32 .sat clamps positive over",
          "[cvt][strategy][f2i][sat]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s32();
    ctx.has_sat = true;
    int32_t dst = 0;
    float src = 1e10f; // > INT32_MAX
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 2147483647);
}

TEST_CASE("FloatToInt f32->s32 .sat clamps negative under",
          "[cvt][strategy][f2i][sat]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s32();
    ctx.has_sat = true;
    int32_t dst = 0;
    float src = -1e10f;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == -2147483647 - 1);
}

// ---- 5 rounding modes (.rn/.rz/.rm/.rp/.rna) ----

TEST_CASE("FloatToInt f32->s32 .rn (round to nearest even)",
          "[cvt][strategy][f2i][round]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s32();
    ctx.has_rn = true;
    int32_t dst = 0;
    float src = 3.5f; // rounds to 4 (even)
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 4);
}

TEST_CASE("FloatToInt f32->s32 .rz (round toward zero)",
          "[cvt][strategy][f2i][round]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s32();
    ctx.has_rz = true;
    int32_t dst = 0;
    float src = -3.7f; // rounds to -3 (toward zero)
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == -3);
}

TEST_CASE("FloatToInt f32->s32 .rm (round toward -inf)",
          "[cvt][strategy][f2i][round]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s32();
    ctx.has_rm = true;
    int32_t dst = 0;
    float src = 3.7f; // rounds to 3 (toward -inf)
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 3);
}

TEST_CASE("FloatToInt f32->s32 .rp (round toward +inf)",
          "[cvt][strategy][f2i][round]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s32();
    ctx.has_rp = true;
    int32_t dst = 0;
    float src = 3.2f; // rounds to 4 (toward +inf)
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 4);
}

TEST_CASE("FloatToInt f32->s32 .rna (round away from zero)",
          "[cvt][strategy][f2i][round]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s32();
    ctx.has_rna = true;
    int32_t dst = 0;
    float src = 3.5f; // rounds to 4 (away from zero)
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 4);
}

TEST_CASE("FloatToInt f32->s32 .rni (alias for rn)",
          "[cvt][strategy][f2i][round]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s32();
    ctx.has_rni = true; // int rounding alias for float
    int32_t dst = 0;
    float src = 3.5f;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 4);
}

TEST_CASE("FloatToInt f32->s32 default truncation",
          "[cvt][strategy][f2i][default]") {
    // No rounding qualifier, no .sat → default behavior is truncation (PTX
    // spec)
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s32();
    int32_t dst = 0;
    float src = 3.7f;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 3);
}

TEST_CASE("FloatToInt f32->s16 .sat overflow clamp",
          "[cvt][strategy][f2i][sat]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s16();
    ctx.has_sat = true;
    int16_t dst = 0;
    float src = 50000.0f; // > 32767
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 32767);
}

TEST_CASE("FloatToInt f32->u16 .sat negative -> 0",
          "[cvt][strategy][f2i][sat]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_u16();
    ctx.has_sat = true;
    uint16_t dst = 0xFFFF;
    float src = -100.0f;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 0);
}

TEST_CASE("FloatToInt f64->s64 .sat basic", "[cvt][strategy][f2i][sat]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s64();
    ctx.has_sat = true;
    int64_t dst = 0;
    double src = 1e20; // > INT64_MAX
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 9223372036854775807LL);
}

TEST_CASE("FloatToInt f64->u64 .sat boundary", "[cvt][strategy][f2i][sat]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_u64();
    ctx.has_sat = true;
    uint64_t dst = 0;
    double src = 1e20; // > UINT64_MAX
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 18446744073709551615ULL);
}

TEST_CASE("FloatToInt f16->s8 .sat (half precision)",
          "[cvt][strategy][f2i][half]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s8();
    ctx.src_is_half = true;
    ctx.src_bytes = 2;
    ctx.has_sat = true;
    int8_t dst = 0;
    uint16_t src = 0x5640; // 100.0 in half
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 100);
}

TEST_CASE("FloatToInt f16->s8 .sat NaN (half)", "[cvt][strategy][f2i][half]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s8();
    ctx.src_is_half = true;
    ctx.src_bytes = 2;
    ctx.has_sat = true;
    int8_t dst = -1;
    uint16_t src = 0x7E00; // NaN in half
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 0);
}

TEST_CASE("FloatToInt f64->s32 .rn", "[cvt][strategy][f2i][round]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s32();
    ctx.src_bytes = 8; // f64 source
    ctx.has_rn = true;
    int32_t dst = 0;
    double src = 3.5;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 4);
}

TEST_CASE("FloatToInt f32->s32 .rp boundary to 0",
          "[cvt][strategy][f2i][round]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s32();
    ctx.has_rp = true;
    int32_t dst = 999;
    float src = 0.1f; // rounds up to 1
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 1);
}

TEST_CASE("FloatToInt f32->u32 .rz truncates positive",
          "[cvt][strategy][f2i][round]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_u32();
    ctx.has_rz = true;
    uint32_t dst = 0;
    float src = 100.9f;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 100);
}

TEST_CASE("FloatToInt f32->s8 default truncation passthrough",
          "[cvt][strategy][f2i][default]") {
    FloatToIntStrategy s;
    auto ctx = make_f2i_dst_s8();
    int8_t dst = 0;
    float src = 5.0f; // exact int
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 5);
}
