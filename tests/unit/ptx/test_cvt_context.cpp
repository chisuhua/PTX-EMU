// test_cvt_context.cpp
// =============================================================================
// Unit test (类型一): 验证 CvtContext + build_context() 的 qualifiers 提取逻辑
//
// 背景:
//   T2-6 Sub-task 3 — CVT 策略模式重构的骨架阶段。
//   build_context() 把 Qualifier 列表解析为强类型字段，
//   供 select_strategy() 选择具体 strategy 使用。
//
// 涵盖 12 类 qualifier 组合:
//   - dst_bytes / src_bytes (1/2/4/8)
//   - dst_is_float / src_is_float
//   - dst_is_half / src_is_half
//   - dst_is_signed / src_is_signed
//   - has_sat / has_rn / has_rni / has_rz / has_rzi
//   - has_rm / has_rmi / has_rp / has_rpi / has_rna / has_rs
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/instructions/cvt/cvt_strategy.h"

#include <vector>

using ptxsim::cvt_strategy::build_context;
using ptxsim::cvt_strategy::CvtContext;

namespace {

// 构造一个只含 dst/src data qualifier 的 Qualifier 列表 (无 .sat / .rni
// 等修饰符)
std::vector<Qualifier> cvt_quals(Qualifier dst_dtype, Qualifier src_dtype) {
    return {dst_dtype, src_dtype};
}

// 构造带 .sat 修饰符的 Qualifier 列表
std::vector<Qualifier> cvt_quals_sat(Qualifier dst_dtype, Qualifier src_dtype) {
    return {dst_dtype, src_dtype, Qualifier::Q_SAT};
}

// 构造带指定 rounding modifier 的 Qualifier 列表
std::vector<Qualifier> cvt_quals_round(Qualifier dst_dtype, Qualifier src_dtype,
                                       Qualifier rmod) {
    return {dst_dtype, src_dtype, rmod};
}

} // namespace

TEST_CASE("build_context basic type extraction", "[cvt][context][types]") {
    SECTION("f32 -> s32: src float, dst signed int") {
        auto ctx = build_context(cvt_quals(Qualifier::Q_S32, Qualifier::Q_F32));
        REQUIRE(ctx.dst_bytes == 4);
        REQUIRE(ctx.src_bytes == 4);
        REQUIRE(ctx.dst_is_float == false);
        REQUIRE(ctx.src_is_float == true);
        REQUIRE(ctx.dst_is_half == false);
        REQUIRE(ctx.src_is_half == false);
        REQUIRE(ctx.dst_is_signed == true);
        REQUIRE(ctx.src_is_signed ==
                false); // f32 is "unsigned" by TypeUtils convention
        REQUIRE_FALSE(ctx.has_sat);
        REQUIRE_FALSE(ctx.has_rn);
        REQUIRE_FALSE(ctx.has_rni);
    }

    SECTION("f64 -> f32: both floats, src double") {
        auto ctx = build_context(cvt_quals(Qualifier::Q_F32, Qualifier::Q_F64));
        REQUIRE(ctx.dst_bytes == 4);
        REQUIRE(ctx.src_bytes == 8);
        REQUIRE(ctx.dst_is_float == true);
        REQUIRE(ctx.src_is_float == true);
        REQUIRE_FALSE(ctx.dst_is_half);
        REQUIRE_FALSE(ctx.src_is_half);
    }

    SECTION("s16 -> s32: both ints, signed extend") {
        auto ctx = build_context(cvt_quals(Qualifier::Q_S32, Qualifier::Q_S16));
        REQUIRE(ctx.dst_bytes == 4);
        REQUIRE(ctx.src_bytes == 2);
        REQUIRE_FALSE(ctx.dst_is_float);
        REQUIRE_FALSE(ctx.src_is_float);
        REQUIRE(ctx.dst_is_signed == true);
        REQUIRE(ctx.src_is_signed == true);
    }

    SECTION("u8 -> u64: unsigned widen") {
        auto ctx = build_context(cvt_quals(Qualifier::Q_U64, Qualifier::Q_U8));
        REQUIRE(ctx.dst_bytes == 8);
        REQUIRE(ctx.src_bytes == 1);
        REQUIRE(ctx.dst_is_signed == false);
        REQUIRE(ctx.src_is_signed == false);
    }
}

TEST_CASE("build_context half (f16) is treated as 2-byte float",
          "[cvt][context][half]") {
    SECTION("f16 -> f32") {
        auto ctx = build_context(cvt_quals(Qualifier::Q_F32, Qualifier::Q_F16));
        REQUIRE(ctx.src_bytes == 2);       // Q_F16 forces 2-byte width
        REQUIRE(ctx.src_is_float == true); // half is float
        REQUIRE(ctx.src_is_half == true);
        REQUIRE(ctx.dst_bytes == 4);
        REQUIRE(ctx.dst_is_float == true);
    }

    SECTION("f32 -> f16") {
        auto ctx = build_context(cvt_quals(Qualifier::Q_F16, Qualifier::Q_F32));
        REQUIRE(ctx.dst_bytes == 2);
        REQUIRE(ctx.dst_is_float == true); // f16 is float
        REQUIRE(ctx.dst_is_half == true);
        REQUIRE(ctx.src_bytes == 4);
        REQUIRE(ctx.src_is_float == true);
    }

    SECTION("f16 -> s16 (half to int)") {
        auto ctx = build_context(cvt_quals(Qualifier::Q_S16, Qualifier::Q_F16));
        REQUIRE(ctx.src_bytes == 2);
        REQUIRE(ctx.src_is_float == true);
        REQUIRE(ctx.src_is_half == true);
        REQUIRE(ctx.dst_bytes == 2);
        REQUIRE(ctx.dst_is_float == false);
        REQUIRE(ctx.dst_is_signed == true);
    }
}

TEST_CASE("build_context .sat qualifier", "[cvt][context][sat]") {
    auto ctx = build_context(cvt_quals_sat(Qualifier::Q_U32, Qualifier::Q_F32));
    REQUIRE(ctx.has_sat);
    REQUIRE(ctx.dst_is_float == false);
    REQUIRE(ctx.src_is_float == true);
}

TEST_CASE("build_context float rounding qualifiers (.rn/.rz/.rm/.rp/.rna)",
          "[cvt][context][rounding]") {
    SECTION(".rn") {
        auto ctx = build_context(cvt_quals_round(
            Qualifier::Q_S32, Qualifier::Q_F32, Qualifier::Q_RN));
        REQUIRE(ctx.has_rn);
        REQUIRE_FALSE(ctx.has_rni);
        REQUIRE_FALSE(ctx.has_rz);
    }

    SECTION(".rz") {
        auto ctx = build_context(cvt_quals_round(
            Qualifier::Q_S32, Qualifier::Q_F32, Qualifier::Q_RZ));
        REQUIRE(ctx.has_rz);
        REQUIRE_FALSE(ctx.has_rn);
    }

    SECTION(".rm") {
        auto ctx = build_context(cvt_quals_round(
            Qualifier::Q_S32, Qualifier::Q_F32, Qualifier::Q_RM));
        REQUIRE(ctx.has_rm);
        REQUIRE_FALSE(ctx.has_rp);
    }

    SECTION(".rp") {
        auto ctx = build_context(cvt_quals_round(
            Qualifier::Q_S32, Qualifier::Q_F32, Qualifier::Q_RP));
        REQUIRE(ctx.has_rp);
        REQUIRE_FALSE(ctx.has_rm);
    }

    SECTION(".rna") {
        auto ctx = build_context(cvt_quals_round(
            Qualifier::Q_S32, Qualifier::Q_F32, Qualifier::Q_RNA));
        REQUIRE(ctx.has_rna);
        REQUIRE_FALSE(ctx.has_rn);
    }
}

TEST_CASE("build_context integer rounding qualifiers (.rni/.rzi/.rmi/.rpi)",
          "[cvt][context][rounding][int]") {
    SECTION(".rni") {
        auto ctx = build_context(cvt_quals_round(
            Qualifier::Q_S32, Qualifier::Q_F32, Qualifier::Q_RNI));
        REQUIRE(ctx.has_rni);
        REQUIRE_FALSE(ctx.has_rn);
    }

    SECTION(".rzi") {
        auto ctx = build_context(cvt_quals_round(
            Qualifier::Q_S32, Qualifier::Q_F32, Qualifier::Q_RZI));
        REQUIRE(ctx.has_rzi);
        REQUIRE_FALSE(ctx.has_rz);
    }

    SECTION(".rmi") {
        auto ctx = build_context(cvt_quals_round(
            Qualifier::Q_S32, Qualifier::Q_F32, Qualifier::Q_RMI));
        REQUIRE(ctx.has_rmi);
        REQUIRE_FALSE(ctx.has_rm);
    }

    SECTION(".rpi") {
        auto ctx = build_context(cvt_quals_round(
            Qualifier::Q_S32, Qualifier::Q_F32, Qualifier::Q_RPI));
        REQUIRE(ctx.has_rpi);
        REQUIRE_FALSE(ctx.has_rp);
    }
}

TEST_CASE("build_context .rs stochastic rounding",
          "[cvt][context][rounding][rs]") {
    auto ctx = build_context(
        cvt_quals_round(Qualifier::Q_F32, Qualifier::Q_F32, Qualifier::Q_RS));
    REQUIRE(ctx.has_rs);
    REQUIRE_FALSE(ctx.has_rn);
}

TEST_CASE("build_context signed/unsigned for various types",
          "[cvt][context][signed]") {
    SECTION("s8 is signed") {
        auto ctx = build_context(cvt_quals(Qualifier::Q_S8, Qualifier::Q_S8));
        REQUIRE(ctx.dst_is_signed);
        REQUIRE(ctx.src_is_signed);
        REQUIRE(ctx.dst_bytes == 1);
    }

    SECTION("u16 is unsigned") {
        auto ctx = build_context(cvt_quals(Qualifier::Q_U16, Qualifier::Q_U16));
        REQUIRE_FALSE(ctx.dst_is_signed);
        REQUIRE_FALSE(ctx.src_is_signed);
    }

    SECTION("s64 is signed") {
        auto ctx = build_context(cvt_quals(Qualifier::Q_S64, Qualifier::Q_S64));
        REQUIRE(ctx.dst_is_signed);
        REQUIRE(ctx.src_is_signed);
        REQUIRE(ctx.dst_bytes == 8);
    }

    SECTION("f64 is unsigned (by TypeUtils convention)") {
        // floats are not is_signed_type -> false
        auto ctx = build_context(cvt_quals(Qualifier::Q_F64, Qualifier::Q_F64));
        REQUIRE_FALSE(ctx.dst_is_signed);
        REQUIRE_FALSE(ctx.src_is_signed);
    }
}

TEST_CASE("build_context all bytes sizes 1/2/4/8", "[cvt][context][bytes]") {
    // 1-byte
    REQUIRE(
        build_context(cvt_quals(Qualifier::Q_U8, Qualifier::Q_U8)).dst_bytes ==
        1);
    // 2-byte
    REQUIRE(build_context(cvt_quals(Qualifier::Q_U16, Qualifier::Q_U16))
                .dst_bytes == 2);
    // 4-byte
    REQUIRE(build_context(cvt_quals(Qualifier::Q_U32, Qualifier::Q_U32))
                .dst_bytes == 4);
    // 8-byte
    REQUIRE(build_context(cvt_quals(Qualifier::Q_U64, Qualifier::Q_U64))
                .dst_bytes == 8);
}

TEST_CASE("build_context signedness by CVT rules",
          "[cvt][context][float_path]") {
    // When dst is float, signedness flag comes from src only for the float path
    // This test just confirms signedness is extracted correctly for float ops
    SECTION("s32 -> f32") {
        auto ctx = build_context(cvt_quals(Qualifier::Q_F32, Qualifier::Q_S32));
        REQUIRE(ctx.dst_is_float);
        REQUIRE(ctx.dst_is_signed == false);
        REQUIRE(ctx.src_is_signed);
    }

    SECTION("u32 -> f32") {
        auto ctx = build_context(cvt_quals(Qualifier::Q_F32, Qualifier::Q_U32));
        REQUIRE(ctx.dst_is_float);
        REQUIRE_FALSE(ctx.dst_is_signed);
        REQUIRE_FALSE(ctx.src_is_signed);
    }
}
