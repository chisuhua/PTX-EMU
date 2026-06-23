// test_cvt_float_to_float.cpp
// =============================================================================
// Unit test (类型一): FloatToFloatStrategy 行为锁定
//
// 背景:
//   T2-6 Sub-task 4a — CVT 策略模式拆分第一个具体策略。
//   FloatToFloatStrategy 处理: dst_is_float && src_is_float
//   (f16/f32/f64 之间的相互转换, 含 .sat 处理 NaN→0)
//
// 行为锁定目的:
//   从 GeneralCvtStrategy::convert() 抽出此子树的 switch 逻辑后,
//   这些测试保证行为一致；Sub-task 6 删除 GeneralCvtStrategy 时不破坏。
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/instructions/cvt/cvt_float_to_float.h"
#include "ptxsim/instructions/cvt/cvt_strategy.h"
#include "ptxsim/utils/half_utils.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

using ptxsim::cvt_strategy::CvtContext;
using ptxsim::cvt_strategy::FloatToFloatStrategy;

namespace {

// 构造 f32→f32 上下文 (单位转换)
CvtContext make_ctx_f32_f32() {
    CvtContext ctx;
    ctx.dst_bytes = 4;
    ctx.src_bytes = 4;
    ctx.dst_is_float = true;
    ctx.src_is_float = true;
    return ctx;
}

CvtContext make_ctx_f32_f64() {
    CvtContext ctx;
    ctx.dst_bytes = 4;
    ctx.src_bytes = 8;
    ctx.dst_is_float = true;
    ctx.src_is_float = true;
    return ctx;
}

CvtContext make_ctx_f64_f32() {
    CvtContext ctx;
    ctx.dst_bytes = 8;
    ctx.src_bytes = 4;
    ctx.dst_is_float = true;
    ctx.src_is_float = true;
    return ctx;
}

CvtContext make_ctx_f64_f64() {
    CvtContext ctx;
    ctx.dst_bytes = 8;
    ctx.src_bytes = 8;
    ctx.dst_is_float = true;
    ctx.src_is_float = true;
    return ctx;
}

CvtContext make_ctx_f32_f16() {
    CvtContext ctx;
    ctx.dst_bytes = 4;
    ctx.src_bytes = 2;
    ctx.dst_is_float = true;
    ctx.src_is_float = true;
    ctx.src_is_half = true;
    return ctx;
}

CvtContext make_ctx_f16_f32() {
    CvtContext ctx;
    ctx.dst_bytes = 2;
    ctx.src_bytes = 4;
    ctx.dst_is_float = true;
    ctx.src_is_float = true;
    ctx.dst_is_half = true;
    return ctx;
}

CvtContext make_ctx_f16_f16() {
    CvtContext ctx;
    ctx.dst_bytes = 2;
    ctx.src_bytes = 2;
    ctx.dst_is_float = true;
    ctx.src_is_float = true;
    ctx.dst_is_half = true;
    ctx.src_is_half = true;
    return ctx;
}

// 工具: float → bit pattern (用于 dst 写入后回读)
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
uint16_t f16_bits(uint16_t h) { return h; }

} // namespace

TEST_CASE("FloatToFloat f32->f32 identity", "[cvt][strategy][f2f][identity]") {
    FloatToFloatStrategy s;
    float dst = 0.0f;
    float src = 3.14159f;
    s.convert(&dst, &src, make_ctx_f32_f32());
    REQUIRE(dst == 3.14159f);
    REQUIRE(f32_bits(dst) == f32_bits(3.14159f));
}

TEST_CASE("FloatToFloat f32->f32 .sat handles NaN -> 0",
          "[cvt][strategy][f2f][sat]") {
    FloatToFloatStrategy s;
    auto ctx = make_ctx_f32_f32();
    ctx.has_sat = true;
    float dst = 999.0f;
    float src = std::numeric_limits<float>::quiet_NaN();
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 0.0f);
}

TEST_CASE("FloatToFloat f32->f32 .sat non-NaN passes through",
          "[cvt][strategy][f2f][sat]") {
    FloatToFloatStrategy s;
    auto ctx = make_ctx_f32_f32();
    ctx.has_sat = true;
    float dst = 0.0f;
    float src = -2.5f;
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == -2.5f);
}

TEST_CASE("FloatToFloat f64->f32 narrowing", "[cvt][strategy][f2f][narrow]") {
    FloatToFloatStrategy s;
    float dst = 0.0f;
    double src = 1.0 / 3.0; // 0.3333333...
    s.convert(&dst, &src, make_ctx_f32_f64());
    // Convert to float for expected (round-trip)
    float expected = static_cast<float>(src);
    REQUIRE(dst == expected);
}

TEST_CASE("FloatToFloat f32->f64 widening", "[cvt][strategy][f2f][widen]") {
    FloatToFloatStrategy s;
    double dst = 0.0;
    float src = 3.14159f;
    s.convert(&dst, &src, make_ctx_f64_f32());
    REQUIRE(dst == static_cast<double>(3.14159f));
    REQUIRE(f64_bits(dst) == f64_bits(static_cast<double>(3.14159f)));
}

TEST_CASE("FloatToFloat f64->f64 identity", "[cvt][strategy][f2f][identity]") {
    FloatToFloatStrategy s;
    double dst = 0.0;
    double src = 2.718281828459045;
    s.convert(&dst, &src, make_ctx_f64_f64());
    REQUIRE(dst == 2.718281828459045);
}

TEST_CASE("FloatToFloat f16->f32 via half_to_float",
          "[cvt][strategy][f2f][half]") {
    FloatToFloatStrategy s;
    float dst = 0.0f;
    uint16_t src = 0x3C00; // 1.0 in half
    s.convert(&dst, &src, make_ctx_f32_f16());
    REQUIRE(dst == 1.0f);
}

TEST_CASE("FloatToFloat f16->f32 half denormal", "[cvt][strategy][f2f][half]") {
    // Smallest positive denormal half: 0x0001 = 2^-24
    FloatToFloatStrategy s;
    float dst = 0.0f;
    uint16_t src = 0x0001;
    s.convert(&dst, &src, make_ctx_f32_f16());
    REQUIRE(dst > 0.0f);
    REQUIRE(dst < 1e-7f);
    REQUIRE(f32_bits(dst) == f32_bits(5.9604644775390625e-08f));
}

TEST_CASE("FloatToFloat f32->f16 via float_to_half",
          "[cvt][strategy][f2f][half]") {
    FloatToFloatStrategy s;
    uint16_t dst = 0;
    float src = 1.0f;
    s.convert(&dst, &src, make_ctx_f16_f32());
    // 1.0 in half = 0x3C00
    REQUIRE(dst == 0x3C00);
}

TEST_CASE("FloatToFloat f16->f16 identity", "[cvt][strategy][f2f][half]") {
    FloatToFloatStrategy s;
    uint16_t dst = 0;
    uint16_t src = 0x4248; // 3.140625 in half (representable)
    s.convert(&dst, &src, make_ctx_f16_f16());
    REQUIRE(dst == 0x4248);
}

TEST_CASE("FloatToFloat f16->f32 inf handling", "[cvt][strategy][f2f][half]") {
    FloatToFloatStrategy s;
    float dst = 0.0f;
    uint16_t src = 0x7C00; // +inf in half
    s.convert(&dst, &src, make_ctx_f32_f16());
    REQUIRE(std::isinf(dst));
    REQUIRE(dst > 0.0f);
}

TEST_CASE("FloatToFloat f32->f16 inf handling", "[cvt][strategy][f2f][half]") {
    FloatToFloatStrategy s;
    uint16_t dst = 0;
    float src = std::numeric_limits<float>::infinity();
    s.convert(&dst, &src, make_ctx_f16_f32());
    // +inf in half = 0x7C00
    REQUIRE(dst == 0x7C00);
}

TEST_CASE("FloatToFloat f32->f16 NaN handling", "[cvt][strategy][f2f][half]") {
    FloatToFloatStrategy s;
    uint16_t dst = 0;
    float src = std::numeric_limits<float>::quiet_NaN();
    s.convert(&dst, &src, make_ctx_f16_f32());
    // NaN in half = 0x7E00
    REQUIRE(dst == 0x7E00);
}

TEST_CASE("FloatToFloat f64<-f16 widens via half->float->double",
          "[cvt][strategy][f2f][half]") {
    CvtContext ctx;
    ctx.dst_bytes = 8;
    ctx.src_bytes = 2;
    ctx.dst_is_float = true;
    ctx.src_is_float = true;
    ctx.src_is_half = true;

    FloatToFloatStrategy s;
    double dst = 0.0;
    uint16_t src = 0x4000; // 2.0 in half
    s.convert(&dst, &src, ctx);
    REQUIRE(dst == 2.0);
}
