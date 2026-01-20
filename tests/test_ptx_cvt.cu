#include "catch_amalgamated.hpp"
#include "ptx_cvt_arith.cuh"
#include <cmath>
#include <cstdint>

// Helper function to compare floats with tolerance
bool float_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) <= epsilon;
}

bool double_equal(double a, double b, double epsilon = 1e-10) {
    return std::abs(a - b) <= epsilon;
}

// Integer to integer conversions
TEST_CASE("PTX: cvt.s8.s16", "[ptx][cvt][integer][s8_s16]") {
    int8_t result;
    test_ptx_cvt_s8_s16(100, &result);
    REQUIRE(result == 100);
}

TEST_CASE("PTX: cvt.s8.s32", "[ptx][cvt][integer][s8_s32]") {
    int8_t result;
    test_ptx_cvt_s8_s32(42, &result);
    REQUIRE(result == 42);
}

TEST_CASE("PTX: cvt.s8.s64", "[ptx][cvt][integer][s8_s64]") {
    int8_t result;
    test_ptx_cvt_s8_s64(127, &result);
    REQUIRE(result == 127);
}

TEST_CASE("PTX: cvt.u8.u16", "[ptx][cvt][integer][u8_u16]") {
    uint8_t result;
    test_ptx_cvt_u8_u16(200, &result);
    REQUIRE(result == 200);
}

TEST_CASE("PTX: cvt.u8.u32", "[ptx][cvt][integer][u8_u32]") {
    uint8_t result;
    test_ptx_cvt_u8_u32(255, &result);
    REQUIRE(result == 255);
}

TEST_CASE("PTX: cvt.u8.u64", "[ptx][cvt][integer][u8_u64]") {
    uint8_t result;
    test_ptx_cvt_u8_u64(100, &result);
    REQUIRE(result == 100);
}

TEST_CASE("PTX: cvt.s16.s8", "[ptx][cvt][integer][s16_s8]") {
    int16_t result;
    test_ptx_cvt_s16_s8(100, &result);
    REQUIRE(result == 100);
}

TEST_CASE("PTX: cvt.s16.s32", "[ptx][cvt][integer][s16_s32]") {
    int16_t result;
    test_ptx_cvt_s16_s32(32000, &result);
    REQUIRE(result == 32000);
}

TEST_CASE("PTX: cvt.s16.s64", "[ptx][cvt][integer][s16_s64]") {
    int16_t result;
    test_ptx_cvt_s16_s64(-5000, &result);
    REQUIRE(result == -5000);
}

TEST_CASE("PTX: cvt.u16.u8", "[ptx][cvt][integer][u16_u8]") {
    uint16_t result;
    test_ptx_cvt_u16_u8(200, &result);
    REQUIRE(result == 200);
}

TEST_CASE("PTX: cvt.u16.u32", "[ptx][cvt][integer][u16_u32]") {
    uint16_t result;
    test_ptx_cvt_u16_u32(60000, &result);
    REQUIRE(result == 60000);
}

TEST_CASE("PTX: cvt.u16.u64", "[ptx][cvt][integer][u16_u64]") {
    uint16_t result;
    test_ptx_cvt_u16_u64(50000, &result);
    REQUIRE(result == 50000);
}

TEST_CASE("PTX: cvt.s32.s8", "[ptx][cvt][integer][s32_s8]") {
    int32_t result;
    test_ptx_cvt_s32_s8(100, &result);
    REQUIRE(result == 100);
}

TEST_CASE("PTX: cvt.s32.s16", "[ptx][cvt][integer][s32_s16]") {
    int32_t result;
    test_ptx_cvt_s32_s16(20000, &result);
    REQUIRE(result == 20000);
}

TEST_CASE("PTX: cvt.s32.s64", "[ptx][cvt][integer][s32_s64]") {
    int32_t result;
    test_ptx_cvt_s32_s64(-100000, &result);
    REQUIRE(result == -100000);
}

TEST_CASE("PTX: cvt.u32.u8", "[ptx][cvt][integer][u32_u8]") {
    uint32_t result;
    test_ptx_cvt_u32_u8(200, &result);
    REQUIRE(result == 200);
}

TEST_CASE("PTX: cvt.u32.u16", "[ptx][cvt][integer][u32_u16]") {
    uint32_t result;
    test_ptx_cvt_u32_u16(50000, &result);
    REQUIRE(result == 50000);
}

TEST_CASE("PTX: cvt.u32.u64", "[ptx][cvt][integer][u32_u64]") {
    uint32_t result;
    test_ptx_cvt_u32_u64(1000000, &result);
    REQUIRE(result == 1000000);
}

TEST_CASE("PTX: cvt.s64.s8", "[ptx][cvt][integer][s64_s8]") {
    int64_t result;
    test_ptx_cvt_s64_s8(100, &result);
    REQUIRE(result == 100);
}

TEST_CASE("PTX: cvt.s64.s16", "[ptx][cvt][integer][s64_s16]") {
    int64_t result;
    test_ptx_cvt_s64_s16(30000, &result);
    REQUIRE(result == 30000);
}

TEST_CASE("PTX: cvt.s64.s32", "[ptx][cvt][integer][s64_s32]") {
    int64_t result;
    test_ptx_cvt_s64_s32(-1000000, &result);
    REQUIRE(result == -1000000);
}

TEST_CASE("PTX: cvt.u64.u8", "[ptx][cvt][integer][u64_u8]") {
    uint64_t result;
    test_ptx_cvt_u64_u8(200, &result);
    REQUIRE(result == 200);
}

TEST_CASE("PTX: cvt.u64.u16", "[ptx][cvt][integer][u64_u16]") {
    uint64_t result;
    test_ptx_cvt_u64_u16(50000, &result);
    REQUIRE(result == 50000);
}

TEST_CASE("PTX: cvt.u64.u32", "[ptx][cvt][integer][u64_u32]") {
    uint64_t result;
    test_ptx_cvt_u64_u32(1000000, &result);
    REQUIRE(result == 1000000);
}

// Float to integer conversions
TEST_CASE("PTX: cvt.rni.s32.f32", "[ptx][cvt][float_to_int][f32_s32]") {
    int32_t result;
    test_ptx_cvt_s32_f32(42.7f, &result);
    REQUIRE(result == 43);  // Rounded to nearest integer
}

TEST_CASE("PTX: cvt.rni.u32.f32", "[ptx][cvt][float_to_int][f32_u32]") {
    uint32_t result;
    test_ptx_cvt_u32_f32(123.4f, &result);
    REQUIRE(result == 123);  // Rounded to nearest integer
}

TEST_CASE("PTX: cvt.rni.s64.f64", "[ptx][cvt][float_to_int][f64_s64]") {
    int64_t result;
    test_ptx_cvt_s64_f64(123456.789, &result);
    REQUIRE(result == 123457);  // Rounded to nearest integer
}

TEST_CASE("PTX: cvt.rni.u64.f64", "[ptx][cvt][float_to_int][f64_u64]") {
    uint64_t result;
    test_ptx_cvt_u64_f64(987654.321, &result);
    REQUIRE(result == 987654);  // Rounded to nearest integer
}

TEST_CASE("PTX: cvt.rni.s32.f64", "[ptx][cvt][float_to_int][f64_s32]") {
    int32_t result;
    test_ptx_cvt_s32_f64(54321.9, &result);
    REQUIRE(result == 54322);  // Rounded to nearest integer
}

TEST_CASE("PTX: cvt.rni.u32.f64", "[ptx][cvt][float_to_int][f64_u32]") {
    uint32_t result;
    test_ptx_cvt_u32_f64(98765.4, &result);
    REQUIRE(result == 98765);  // Rounded to nearest integer
}

// Integer to float conversions
TEST_CASE("PTX: cvt.f32.s32", "[ptx][cvt][int_to_float][s32_f32]") {
    float result;
    test_ptx_cvt_f32_s32(42, &result);
    REQUIRE(float_equal(result, 42.0f));
}

TEST_CASE("PTX: cvt.f32.u32", "[ptx][cvt][int_to_float][u32_f32]") {
    float result;
    test_ptx_cvt_f32_u32(123, &result);
    REQUIRE(float_equal(result, 123.0f));
}

TEST_CASE("PTX: cvt.f64.s64", "[ptx][cvt][int_to_float][s64_f64]") {
    double result;
    test_ptx_cvt_f64_s64(123456, &result);
    REQUIRE(double_equal(result, 123456.0));
}

TEST_CASE("PTX: cvt.f64.u64", "[ptx][cvt][int_to_float][u64_f64]") {
    double result;
    test_ptx_cvt_f64_u64(987654, &result);
    REQUIRE(double_equal(result, 987654.0));
}

TEST_CASE("PTX: cvt.f32.s64", "[ptx][cvt][int_to_float][s64_f32]") {
    float result;
    test_ptx_cvt_f32_s64(54321, &result);
    REQUIRE(float_equal(result, 54321.0f));
}

TEST_CASE("PTX: cvt.f32.u64", "[ptx][cvt][int_to_float][u64_f32]") {
    float result;
    test_ptx_cvt_f32_u64(98765, &result);
    REQUIRE(float_equal(result, 98765.0f));
}

// Float to float conversions
TEST_CASE("PTX: cvt.f32.f64", "[ptx][cvt][float_to_float][f64_f32]") {
    float result;
    test_ptx_cvt_f32_f64(123.456, &result);
    REQUIRE(float_equal(result, 123.456f));
}

TEST_CASE("PTX: cvt.f64.f32", "[ptx][cvt][float_to_float][f32_f64]") {
    double result;
    test_ptx_cvt_f64_f32(654.321f, &result);
    REQUIRE(double_equal(result, 654.321));
}

// Saturation conversions
TEST_CASE("PTX: cvt.sat.u8.f32", "[ptx][cvt][saturation][f32_u8]") {
    uint8_t result;
    test_ptx_cvt_satu8_f32(260.0f, &result);  // Should saturate to 255
    REQUIRE(result == 255);
}

TEST_CASE("PTX: cvt.sat.u8.f32 negative", "[ptx][cvt][saturation][f32_u8_negative]") {
    uint8_t result;
    test_ptx_cvt_satu8_f32(-5.0f, &result);  // Should saturate to 0
    REQUIRE(result == 0);
}

TEST_CASE("PTX: cvt.sat.u16.f32", "[ptx][cvt][saturation][f32_u16]") {
    uint16_t result;
    test_ptx_cvt_satu16_f32(70000.0f, &result);  // Should saturate to 65535
    REQUIRE(result == 65535);
}

TEST_CASE("PTX: cvt.sat.u32.f32", "[ptx][cvt][saturation][f32_u32]") {
    uint32_t result;
    test_ptx_cvt_satu32_f32(5e9f, &result);  // Should saturate to 4294967295U
    REQUIRE(result == 4294967295U);
}