#include "catch_amalgamated.hpp"
#include "ptx_integer_arith.cuh"
#include <cstdint>

TEST_CASE("PTX: add.s32", "[ptx][integer][add]") {
    int result;
    test_ptx_add_s32(5, 7, &result);
    REQUIRE(result == 12);
}

TEST_CASE("PTX: sub.s32", "[ptx][integer][sub]") {
    int result;
    test_ptx_sub_s32(10, 3, &result);
    REQUIRE(result == 7);
}

TEST_CASE("PTX: mul.lo.s32", "[ptx][integer][mul]") {
    int result;
    test_ptx_mul_s32(6, 7, &result);
    REQUIRE(result == 42);
}

TEST_CASE("PTX: mul24.lo.s32", "[ptx][integer][mul24]") {
    int result;
    test_ptx_mul24_s32(0x7F, 0x7F, &result); // 127*127 = 16129 < 2^24
    REQUIRE(result == 16129);
}

TEST_CASE("PTX: mad.lo.s32", "[ptx][integer][mad]") {
    int result;
    test_ptx_mad_s32(3, 4, 5, &result); // 3*4 + 5 = 17
    REQUIRE(result == 17);
}

TEST_CASE("PTX: mad24.lo.s32", "[ptx][integer][mad24]") {
    int result;
    test_ptx_mad24_s32(10, 20, 5, &result); // 10*20 + 5 = 205
    REQUIRE(result == 205);
}

TEST_CASE("PTX: div.s32", "[ptx][integer][div]") {
    int result;
    test_ptx_div_s32(15, 3, &result);
    REQUIRE(result == 5);
}

TEST_CASE("PTX: rem.s32", "[ptx][integer][rem]") {
    int result;
    test_ptx_rem_s32(17, 5, &result);
    REQUIRE(result == 2);
}

TEST_CASE("PTX: abs.s32", "[ptx][integer][abs]") {
    int result;
    test_ptx_abs_s32(-42, &result);
    REQUIRE(result == 42);
}

TEST_CASE("PTX: neg.s32", "[ptx][integer][neg]") {
    int result;
    test_ptx_neg_s32(100, &result);
    REQUIRE(result == -100);
}

TEST_CASE("PTX: min.s32", "[ptx][integer][min]") {
    int result;
    test_ptx_min_s32(-5, 10, &result);
    REQUIRE(result == -5);
}

TEST_CASE("PTX: max.s32", "[ptx][integer][max]") {
    int result;
    test_ptx_max_s32(-5, 10, &result);
    REQUIRE(result == 10);
}

TEST_CASE("PTX: popc.b32", "[ptx][integer][popc]") {
    int result;
    test_ptx_popc_u32(0b10110100, &result); // 4 bits set
    REQUIRE(result == 4);
}

TEST_CASE("PTX: clz.b32", "[ptx][integer][clz]") {
    int result;
    test_ptx_clz_u32(0x08000000, &result); // 00001000... => 4 leading zeros
    REQUIRE(result == 4);
}

// TEST_CASE("PTX: bfind.reverse.s32", "[ptx][integer][bfind]") {
//     int result;
//     test_ptx_bfind_s32(0b10010000); // Highest set bit at position 7 (0-indexed from LSB)
//     // PTX bfind returns bit index; for 0b10010000 (0x90), highest bit is 7
//     REQUIRE(result == 7);
// }