#include "catch_amalgamated.hpp"
#include "ptx_cvt_arith.h"
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>

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

TEST_CASE("PTX: cvt.s8.s16 overflow", "[ptx][cvt][integer][s8_s16_overflow]") {
    int8_t result;
    test_ptx_cvt_s8_s16(200, &result);  // 200 = 0xC8 -> sign extended as s8 = -56
    REQUIRE(result == -56); // 200 as int8_t is -56
}

TEST_CASE("PTX: cvt.s8.s32", "[ptx][cvt][integer][s8_s32]") {
    int8_t result;
    test_ptx_cvt_s8_s32(42, &result);
    REQUIRE(result == 42);
}

TEST_CASE("PTX: cvt.s8.s32 overflow", "[ptx][cvt][integer][s8_s32_overflow]") {
    int8_t result;
    test_ptx_cvt_s8_s32(300, &result);  // 300 = 0x12C -> low 8 bits = 0x2C = 44
    REQUIRE(result == 44);
}

TEST_CASE("PTX: cvt.s8.s64", "[ptx][cvt][integer][s8_s64]") {
    int8_t result;
    test_ptx_cvt_s8_s64(127, &result);
    REQUIRE(result == 127);
}

TEST_CASE("PTX: cvt.s8.s64 overflow", "[ptx][cvt][integer][s8_s64_overflow]") {
    int8_t result;
    test_ptx_cvt_s8_s64(258, &result);  // 258 = 0x102 -> low 8 bits = 0x02 = 2
    REQUIRE(result == 2);
}

TEST_CASE("PTX: cvt.u8.u16", "[ptx][cvt][integer][u8_u16]") {
    uint8_t result;
    test_ptx_cvt_u8_u16(200, &result);
    REQUIRE(result == 200);
}

TEST_CASE("PTX: cvt.u8.u16 overflow", "[ptx][cvt][integer][u8_u16_overflow]") {
    uint8_t result;
    test_ptx_cvt_u8_u16(300, &result);  // 300 = 0x12C -> low 8 bits = 0x2C = 44
    REQUIRE(result == 44);
}

TEST_CASE("PTX: cvt.u8.u32", "[ptx][cvt][integer][u8_u32]") {
    uint8_t result;
    test_ptx_cvt_u8_u32(255, &result);
    REQUIRE(result == 255);
}

TEST_CASE("PTX: cvt.u8.u32 overflow", "[ptx][cvt][integer][u8_u32_overflow]") {
    uint8_t result;
    test_ptx_cvt_u8_u32(511, &result);  // 511 = 0x1FF -> low 8 bits = 0xFF = 255
    REQUIRE(result == 255);
}

TEST_CASE("PTX: cvt.u8.u64", "[ptx][cvt][integer][u8_u64]") {
    uint8_t result;
    test_ptx_cvt_u8_u64(100, &result);
    REQUIRE(result == 100);
}

TEST_CASE("PTX: cvt.u8.u64 overflow", "[ptx][cvt][integer][u8_u64_overflow]") {
    uint8_t result;
    test_ptx_cvt_u8_u64(355, &result);  // 355 = 0x163 -> low 8 bits = 0x63 = 99
    REQUIRE(result == 99);
}

TEST_CASE("PTX: cvt.s16.s8", "[ptx][cvt][integer][s16_s8]") {
    int16_t result;
    test_ptx_cvt_s16_s8(100, &result);
    REQUIRE(result == 100);
}

TEST_CASE("PTX: cvt.s16.s8 overflow", "[ptx][cvt][integer][s16_s8_overflow]") {
    int16_t result;
    // When converting s8 with value -1 (0xFF) to s16, it should be sign-extended to -1
    test_ptx_cvt_s16_s8(-1, &result);
    REQUIRE(result == -1);
    
    // When converting s8 with value -128 (0x80) to s16, it should be sign-extended to -128
    test_ptx_cvt_s16_s8(-128, &result);
    REQUIRE(result == -128);
    
    // When converting s8 with value 127 (0x7F) to s16, it should remain 127
    test_ptx_cvt_s16_s8(127, &result);
    REQUIRE(result == 127);
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

TEST_CASE("PTX: cvt.u16.u8 overflow", "[ptx][cvt][integer][u16_u8_overflow]") {
    uint16_t result;
    // When u8 value 255 (0xFF) is converted to u16, it should remain 255
    test_ptx_cvt_u16_u8(255, &result);
    REQUIRE(result == 255);
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

TEST_CASE("PTX: cvt.s32.s8 overflow", "[ptx][cvt][integer][s32_s8_overflow]") {
    int32_t result;
    test_ptx_cvt_s32_s8(300, &result);  // 300 = 0x12C -> low 8 bits = 0x2C = 44
    REQUIRE(result == 44);
}

TEST_CASE("PTX: cvt.s32.s16", "[ptx][cvt][integer][s32_s16]") {
    int32_t result;
    test_ptx_cvt_s32_s16(20000, &result);
    REQUIRE(result == 20000);
}

TEST_CASE("PTX: cvt.s32.s16 overflow", "[ptx][cvt][integer][s32_s16_overflow]") {
    int32_t result;
    test_ptx_cvt_s32_s16(70000, &result);  // 70000 = 0x11170 -> low 16 bits = 0x1170 = 4464
    REQUIRE(result == 4464);
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

TEST_CASE("PTX: cvt.u32.u8 overflow", "[ptx][cvt][integer][u32_u8_overflow]") {
    uint32_t result;
    test_ptx_cvt_u32_u8(600, &result);  // 600 = 0x258 -> low 8 bits = 0x58 = 88
    REQUIRE(result == 88);
}

TEST_CASE("PTX: cvt.u32.u16", "[ptx][cvt][integer][u32_u16]") {
    uint32_t result;
    test_ptx_cvt_u32_u16(50000, &result);
    REQUIRE(result == 50000);
}

TEST_CASE("PTX: cvt.u32.u16 overflow", "[ptx][cvt][integer][u32_u16_overflow]") {
    uint32_t result;
    test_ptx_cvt_u32_u16(70000, &result);  // 70000 = 0x11170 -> low 16 bits = 0x1170 = 4464
    REQUIRE(result == 4464);
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

TEST_CASE("PTX: cvt.s64.s8 overflow", "[ptx][cvt][integer][s64_s8_overflow]") {
    int64_t result;
    test_ptx_cvt_s64_s8(300, &result);  // 300 = 0x12C -> low 8 bits = 0x2C = 44
    REQUIRE(result == 44);
}

TEST_CASE("PTX: cvt.s64.s16", "[ptx][cvt][integer][s64_s16]") {
    int64_t result;
    test_ptx_cvt_s64_s16(30000, &result);
    REQUIRE(result == 30000);
}

TEST_CASE("PTX: cvt.s64.s16 overflow", "[ptx][cvt][integer][s64_s16_overflow]") {
    int64_t result;
    test_ptx_cvt_s64_s16(70000, &result);  // 70000 = 0x11170 -> low 16 bits = 0x1170 = 4464
    REQUIRE(result == 4464);
}

TEST_CASE("PTX: cvt.s64.s32", "[ptx][cvt][integer][s64_s32]") {
    int64_t result;
    test_ptx_cvt_s64_s32(-1000000, &result);
    REQUIRE(result == -1000000);
}

TEST_CASE("PTX: cvt.s64.s32 overflow", "[ptx][cvt][integer][s64_s32_overflow]") {
    int64_t result;
    // Testing with value larger than 32-bit signed integer max
    test_ptx_cvt_s64_s32(4000000000, &result);  // 4000000000 = 0xEE6B2800 -> as s32 = -294967296
    REQUIRE(result == -294967296);
}

TEST_CASE("PTX: cvt.u64.u8", "[ptx][cvt][integer][u64_u8]") {
    uint64_t result;
    test_ptx_cvt_u64_u8(200, &result);
    REQUIRE(result == 200);
}

TEST_CASE("PTX: cvt.u64.u8 overflow", "[ptx][cvt][integer][u64_u8_overflow]") {
    uint64_t result;
    test_ptx_cvt_u64_u8(600, &result);  // 600 = 0x258 -> low 8 bits = 0x58 = 88
    REQUIRE(result == 88);
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

TEST_CASE("PTX: cvt.u64.u32 overflow", "[ptx][cvt][integer][u64_u32_overflow]") {
    uint64_t result;
    // Testing with value larger than 32-bit unsigned integer max
    test_ptx_cvt_u64_u32(5000000000UL, &result);  // 5000000000 = 0x12A05F200 -> low 32 bits = 0x2A05F200 = 705032704
    REQUIRE(result == 705032704);
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

// Half precision (f16) to float (f32) conversions
TEST_CASE("PTX: cvt.f32.f16", "[ptx][cvt][float_to_float][f16_f32]") {
    float result;
    __half input = __float2half(3.14f);
    test_ptx_cvt_f32_f16(input, &result);
    REQUIRE(float_equal(result, 3.14f, 0.01f));  // Allow slightly larger tolerance due to precision loss
}

TEST_CASE("PTX: cvt.f16.f32", "[ptx][cvt][float_to_float][f32_f16]") {
    __half result;
    float input = 2.71f;
    test_ptx_cvt_f16_f32(input, &result);
    float converted_back = __half2float(result);
    REQUIRE(float_equal(converted_back, 2.71f, 0.01f));  // Allow slightly larger tolerance due to precision loss
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