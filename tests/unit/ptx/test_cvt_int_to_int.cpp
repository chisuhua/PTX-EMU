// test_cvt_int_to_int.cpp
// =============================================================================
// Unit test (类型一): IntToIntStrategy 行为锁定
//
// 背景:
//   T2-6 Sub-task 4d — CVT 策略模式拆分第四个具体策略 (最复杂)。
//   IntToIntStrategy 处理: !dst_is_float && !src_is_float
//   4x4 维度矩阵:
//     - dst_bytes ∈ {1, 2, 4, 8}
//     - src_bytes ∈ {1, 2, 4, 8}
//     - (dst_is_signed, src_is_signed) ∈ {ss, su, us, uu}
//   共 4 * 4 * 4 = 64 个转换组合 + .sat / 5 舍入模式变体
//
// 行为锁定目的: Sub-task 6 删除 GeneralCvtStrategy 时不破坏。
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/instructions/cvt/cvt_int_to_int.h"
#include "ptxsim/instructions/cvt/cvt_strategy.h"

#include <cstdint>
#include <cstring>

using ptxsim::cvt_strategy::CvtContext;
using ptxsim::cvt_strategy::IntToIntStrategy;

namespace {

CvtContext make_i2i(int dst_bytes, bool dst_signed, int src_bytes,
                    bool src_signed) {
    CvtContext ctx;
    ctx.dst_bytes = dst_bytes;
    ctx.dst_is_signed = dst_signed;
    ctx.src_bytes = src_bytes;
    ctx.src_is_signed = src_signed;
    return ctx;
}

} // namespace

// ---- Identity & sign extension (1 byte src) ----

TEST_CASE("IntToInt s8->s16 sign extend", "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    int16_t dst = 0;
    int8_t src = -1;
    s.convert(&dst, &src, make_i2i(2, true, 1, true));
    REQUIRE(dst == -1);
    REQUIRE(static_cast<int16_t>(-1) == -1);
}

TEST_CASE("IntToInt u8->u16 zero extend", "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    uint16_t dst = 0;
    uint8_t src = 200;
    s.convert(&dst, &src, make_i2i(2, false, 1, false));
    REQUIRE(dst == 200);
}

TEST_CASE("IntToInt s8->s32 sign extend", "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    int32_t dst = 0;
    int8_t src = -42;
    s.convert(&dst, &src, make_i2i(4, true, 1, true));
    REQUIRE(dst == -42);
}

TEST_CASE("IntToInt u8->s32 widen", "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    int32_t dst = 0;
    uint8_t src = 200;
    s.convert(&dst, &src, make_i2i(4, true, 1, false));
    REQUIRE(dst == 200);
}

// ---- 4-byte src to 1-byte dst (with .sat and rounding) ----

TEST_CASE("IntToInt s32->s8 .sat clamps positive",
          "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(1, true, 4, true);
    ctx.has_sat = true;
    int8_t dst = 0;
    int32_t src = 200; // > 127
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 127);
}

TEST_CASE("IntToInt s32->s8 .sat clamps negative",
          "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(1, true, 4, true);
    ctx.has_sat = true;
    int8_t dst = 0;
    int32_t src = -200; // < -128
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == -128);
}

TEST_CASE("IntToInt u32->s8 .sat clamps positive (saturate to 127)",
          "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(1, true, 4, false);
    ctx.has_sat = true;
    int8_t dst = 0;
    uint32_t src = 200;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 127);
}

TEST_CASE("IntToInt s32->u8 .sat clamps to [0, 255]",
          "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(1, false, 4, true);
    ctx.has_sat = true;
    uint8_t dst = 0;
    int32_t src = -5;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 0);
}

TEST_CASE("IntToInt s32->u8 .sat in-range passthrough",
          "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(1, false, 4, true);
    ctx.has_sat = true;
    uint8_t dst = 0;
    int32_t src = 200;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 200);
}

TEST_CASE("IntToInt s32->s8 .rn rounding", "[cvt][strategy][i2i][round]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(1, true, 4, true);
    ctx.has_rn = true;
    int8_t dst = 0;
    int32_t src = 100;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 100);
}

TEST_CASE("IntToInt s16->s8 .rz truncation", "[cvt][strategy][i2i][round]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(1, true, 2, true);
    ctx.has_rz = true;
    int8_t dst = 0;
    int16_t src = 200; // > 127 — will be truncated to fit int8
    s.convert(&dst, &src, ctx);
    // Without .sat, truncation: 200 in 8 bits = -56
    REQUIRE(dst == -56);
}

TEST_CASE("IntToInt s16->s8 .rm floor", "[cvt][strategy][i2i][round]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(1, true, 2, true);
    ctx.has_rm = true;
    int8_t dst = 0;
    int16_t src = -200; // rounds toward -inf
    s.convert(&dst, &src, ctx);
    // -200 doesn't fit int8 — wraps to 56
    REQUIRE(dst == 56);
}

// ---- 4-byte src to 2-byte dst ----

TEST_CASE("IntToInt s32->s16 truncation", "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    int16_t dst = 0;
    int32_t src = 30000; // fits in int16
    s.convert(&dst, &src, make_i2i(2, true, 4, true));
    REQUIRE(dst == 30000);
}

TEST_CASE("IntToInt s32->s16 .sat clamps", "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(2, true, 4, true);
    ctx.has_sat = true;
    int16_t dst = 0;
    int32_t src = 50000; // > 32767
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 32767);
}

TEST_CASE("IntToInt s32->u16 .sat clamps", "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(2, false, 4, true);
    ctx.has_sat = true;
    uint16_t dst = 0;
    int32_t src = -1;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 0);
}

TEST_CASE("IntToInt s32->s16 default (no rounding)", "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    int16_t dst = 0;
    int32_t src = 1000;
    s.convert(&dst, &src, make_i2i(2, true, 4, true));
    REQUIRE(dst == 1000);
}

// ---- 8-byte src to 4-byte dst ----

TEST_CASE("IntToInt s64->s32 .sat clamps positive",
          "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(4, true, 8, true);
    ctx.has_sat = true;
    int32_t dst = 0;
    int64_t src = 3000000000LL; // > INT32_MAX
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 2147483647);
}

TEST_CASE("IntToInt s64->s32 .sat clamps negative",
          "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(4, true, 8, true);
    ctx.has_sat = true;
    int32_t dst = 0;
    int64_t src = -3000000000LL;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == -2147483647 - 1);
}

TEST_CASE("IntToInt u64->s32 .sat clamps", "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(4, true, 8, false);
    ctx.has_sat = true;
    int32_t dst = 0;
    uint64_t src = 3000000000ULL;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 2147483647);
}

TEST_CASE("IntToInt s64->u32 .sat clamps negative to 0",
          "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(4, false, 8, true);
    ctx.has_sat = true;
    uint32_t dst = 0;
    int64_t src = -1;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 0);
}

TEST_CASE("IntToInt s64->u32 .sat clamps positive to UINT32_MAX",
          "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(4, false, 8, true);
    ctx.has_sat = true;
    uint32_t dst = 0;
    int64_t src = 5000000000LL; // > UINT32_MAX
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 4294967295U);
}

TEST_CASE("IntToInt s64->s32 in-range passthrough", "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    int32_t dst = 0;
    int64_t src = -1234567;
    s.convert(&dst, &src, make_i2i(4, true, 8, true));
    REQUIRE(dst == -1234567);
}

TEST_CASE("IntToInt u64->u32 truncation", "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    uint32_t dst = 0;
    uint64_t src = 0xDEADBEEFULL;
    s.convert(&dst, &src, make_i2i(4, false, 8, false));
    REQUIRE(dst == 0xDEADBEEF);
}

// ---- 8-byte src to 8-byte dst (identity) ----

TEST_CASE("IntToInt s64->s64 identity", "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    int64_t dst = 0;
    int64_t src = -9000000000LL;
    s.convert(&dst, &src, make_i2i(8, true, 8, true));
    REQUIRE(dst == -9000000000LL);
}

TEST_CASE("IntToInt u64->u64 identity", "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    uint64_t dst = 0;
    uint64_t src = 18000000000ULL;
    s.convert(&dst, &src, make_i2i(8, false, 8, false));
    REQUIRE(dst == 18000000000ULL);
}

TEST_CASE("IntToInt s64->u64 sign mismatch truncation",
          "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    uint64_t dst = 0;
    int64_t src = -1;
    s.convert(&dst, &src, make_i2i(8, false, 8, true));
    // No .sat, no rounding: -1 in 64-bit unsigned is 0xFFFFFFFFFFFFFFFF
    REQUIRE(dst == 18446744073709551615ULL);
}

// ---- 2-byte src to 1-byte dst ----

TEST_CASE("IntToInt s16->s8 .sat clamps positive",
          "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(1, true, 2, true);
    ctx.has_sat = true;
    int8_t dst = 0;
    int16_t src = 500;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 127);
}

TEST_CASE("IntToInt u16->u8 .sat clamps", "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(1, false, 2, false);
    ctx.has_sat = true;
    uint8_t dst = 0;
    uint16_t src = 500;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 255);
}

TEST_CASE("IntToInt s16->s8 in-range", "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    int8_t dst = 0;
    int16_t src = 100;
    s.convert(&dst, &src, make_i2i(1, true, 2, true));
    REQUIRE(dst == 100);
}

// ---- 4-byte src to 1-byte dst (no .sat, with .rni) ----

TEST_CASE("IntToInt s32->s8 .rni rounding", "[cvt][strategy][i2i][round]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(1, true, 4, true);
    ctx.has_rni = true;
    int8_t dst = 0;
    int32_t src = 100;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 100);
}

TEST_CASE("IntToInt s32->s8 .rpi rounding up", "[cvt][strategy][i2i][round]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(1, true, 4, true);
    ctx.has_rpi = true;
    int8_t dst = 0;
    int32_t src = 100;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 100);
}

// ---- Same-size conversions (4-byte identity) ----

TEST_CASE("IntToInt s32->s32 identity", "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    int32_t dst = 0;
    int32_t src = -42;
    s.convert(&dst, &src, make_i2i(4, true, 4, true));
    REQUIRE(dst == -42);
}

TEST_CASE("IntToInt u32->u32 identity", "[cvt][strategy][i2i]") {
    IntToIntStrategy s;
    uint32_t dst = 0;
    uint32_t src = 3000000000U;
    s.convert(&dst, &src, make_i2i(4, false, 4, false));
    REQUIRE(dst == 3000000000U);
}

TEST_CASE("IntToInt s32->u32 .sat passthrough", "[cvt][strategy][i2i][sat]") {
    IntToIntStrategy s;
    auto ctx = make_i2i(4, false, 4, true);
    ctx.has_sat = true;
    uint32_t dst = 0;
    int32_t src = -1;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 0);
}
