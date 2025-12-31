#include "catch_amalgamated.hpp"
#include "ptx_extended_prec.cuh"
#include <cstdint>

TEST_CASE("PTX: addc.b32 without carry", "[ptx][extended][addc][b32]") {
    uint32_t result;
    bool carry_out;
    test_ptx_addc_u32(100, 200, false, &result, &carry_out);
    REQUIRE(result == 300);
    REQUIRE(carry_out == false);
}

TEST_CASE("PTX: addc.b32 with carry", "[ptx][extended][addc][b32]") {
    uint32_t result;
    bool carry_out;
    test_ptx_addc_u32(0xFFFFFFFF, 1, true, &result, &carry_out);
    REQUIRE(result == 1);
    REQUIRE(carry_out == true);
}

TEST_CASE("PTX: subc.b32 without borrow", "[ptx][extended][subc][b32]") {
    uint32_t result;
    bool borrow_out;
    test_ptx_subc_u32(200, 100, false, &result, &borrow_out);
    REQUIRE(result == 100);
    REQUIRE(borrow_out == false);
}

TEST_CASE("PTX: subc.b32 with borrow", "[ptx][extended][subc][b32]") {
    uint32_t result;
    bool borrow_out;
    test_ptx_subc_u32(0, 1, true, &result, &borrow_out);
    REQUIRE(result == 0xFFFFFFFF);
    REQUIRE(borrow_out == true);
}

TEST_CASE("PTX: mul24.lo.u32", "[ptx][extended][mul24][lo][u32]") {
    uint32_t result;
    test_ptx_mul24_lo_u32(0x00000FFF, 0x00000FFF, &result);
    REQUIRE(result == 0x00FFF001);
}

TEST_CASE("PTX: mul24.hi.u32", "[ptx][extended][mul24][hi][u32]") {
    uint32_t result;
    test_ptx_mul24_hi_u32(0x00000FFF, 0x00000FFF, &result);
    REQUIRE(result == 0x00000000);
}

TEST_CASE("PTX: mul.lo.u32", "[ptx][extended][mul][lo][u32]") {
    uint32_t result;
    test_ptx_mul_lo_u32(0x12345678, 0x87654321, &result);
    REQUIRE(result == 0x5D7ABB58);
}

TEST_CASE("PTX: mul.hi.u32", "[ptx][extended][mul][hi][u32]") {
    uint32_t result;
    test_ptx_mul_hi_u32(0x12345678, 0x87654321, &result);
    REQUIRE(result == 0x09A32D70);
}

TEST_CASE("PTX: mul.wide.u32", "[ptx][extended][mul][wide][u32]") {
    uint32_t result;
    test_ptx_mul_wide_u32(0x12345678, 0x87654321, &result);
    REQUIRE(result == 0x5D7ABB58);  // Only lower 32 bits returned
}

TEST_CASE("PTX: mul24.lo.s32", "[ptx][extended][mul24][lo][s32]") {
    int32_t result;
    test_ptx_mul24_lo_s32(0x000007FF, 0x000007FF, &result);
    REQUIRE(result == 0x003FE001);
}

TEST_CASE("PTX: mul24.hi.s32", "[ptx][extended][mul24][hi][s32]") {
    int32_t result;
    test_ptx_mul24_hi_s32(0x000007FF, 0x000007FF, &result);
    REQUIRE(result == 0x00000000);
}

TEST_CASE("PTX: mul.lo.s32", "[ptx][extended][mul][lo][s32]") {
    int32_t result;
    test_ptx_mul_lo_s32(0x12345678, 0x87654321, &result);
    REQUIRE(result == 0x5D7ABB58);
}

TEST_CASE("PTX: mul.hi.s32", "[ptx][extended][mul][hi][s32]") {
    int32_t result;
    test_ptx_mul_hi_s32(0x12345678, 0x87654321, &result);
    REQUIRE(result == 0x09A32D70);
}

TEST_CASE("PTX: mul.wide.u32 to u64", "[ptx][extended][mul][wide][u32_to_u64]") {
    uint64_t result;
    test_ptx_mul_wide_u32_to_u64(0x12345678, 0x87654321, &result);
    REQUIRE(result == 0x09A32D705D7ABB58ULL);
}