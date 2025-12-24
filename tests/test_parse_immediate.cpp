#include "ptxsim/utils/qualifier_utils.h"
#include "ptx_ir/ptx_types.h"
#include "catch_amalgamated.hpp"
#include <cstring>
#include <cmath>

TEST_CASE("PTX Immediate Parsing") {
    float f32;
    double f64;
    uint32_t u32;
    uint64_t u64;
    uint16_t u16;
    uint8_t u8;
    uint8_t pred;
    
    SECTION("Float32 parsing - standard hex format") {
        parseImmediate("0f3F800000", Qualifier::Q_F32, &f32);
        REQUIRE(f32 == 1.0f);
    }
    
    SECTION("Float32 parsing - direct hex format") {
        parseImmediate("0x3F800000", Qualifier::Q_F32, &f32);
        REQUIRE(f32 == 1.0f);
    }
    
    SECTION("Float32 parsing - decimal format") {
        parseImmediate("1.0", Qualifier::Q_F32, &f32);
        REQUIRE(f32 == 1.0f);
    }
    
    SECTION("Float32 parsing - negative value") {
        parseImmediate("-1.5", Qualifier::Q_F32, &f32);
        REQUIRE(f32 == -1.5f);
    }
    
    SECTION("Float32 parsing - zero") {
        parseImmediate("0.0", Qualifier::Q_F32, &f32);
        REQUIRE(f32 == 0.0f);
    }
    
    SECTION("Float32 parsing - hex-float format") {
        parseImmediate("0x1.0p0", Qualifier::Q_F32, &f32);
        REQUIRE(f32 == 1.0f);
    }
    
    SECTION("Float32 parsing - hex-float with exponent") {
        parseImmediate("0x1.0p2", Qualifier::Q_F32, &f32);
        REQUIRE(f32 == 4.0f);
    }
    
    SECTION("Float64 parsing - standard hex format") {
        parseImmediate("0d4000000000000000", Qualifier::Q_F64, &f64);
        REQUIRE(f64 == 2.0);
    }
    
    SECTION("Float64 parsing - direct hex format") {
        parseImmediate("0x4000000000000000", Qualifier::Q_F64, &f64);
        REQUIRE(f64 == 2.0);
    }
    
    SECTION("Float64 parsing - decimal format") {
        parseImmediate("3.14159", Qualifier::Q_F64, &f64);
        REQUIRE(f64 == 3.14159);
    }
    
    SECTION("Float64 parsing - negative value") {
        parseImmediate("-2.5", Qualifier::Q_F64, &f64);
        REQUIRE(f64 == -2.5);
    }
    
    SECTION("Float64 parsing - zero") {
        parseImmediate("0.0", Qualifier::Q_F64, &f64);
        REQUIRE(f64 == 0.0);
    }
    
    SECTION("Int64 parsing - positive value") {
        parseImmediate("123456789", Qualifier::Q_S64, &u64);
        REQUIRE(u64 == 123456789);
    }
    
    SECTION("Int64 parsing - negative value") {
        parseImmediate("-987654321", Qualifier::Q_S64, &u64);
        REQUIRE(u64 == static_cast<uint64_t>(-987654321));
    }
    
    SECTION("Int64 parsing - hex value") {
        parseImmediate("0x123456789ABCDEF0", Qualifier::Q_U64, &u64);
        REQUIRE(u64 == 0x123456789ABCDEF0);
    }
    
    SECTION("Int32 parsing - positive value") {
        parseImmediate("12345", Qualifier::Q_S32, &u32);
        REQUIRE(u32 == 12345);
    }
    
    SECTION("Int32 parsing - negative value") {
        parseImmediate("-54321", Qualifier::Q_S32, &u32);
        REQUIRE(u32 == static_cast<uint32_t>(-54321));
    }
    
    SECTION("Int32 parsing - hex value") {
        parseImmediate("0x1234ABCD", Qualifier::Q_U32, &u32);
        REQUIRE(u32 == 0x1234ABCD);
    }
    
    SECTION("Int32 parsing - overflow handling - high bits masked") {
        parseImmediate("0x12345678", Qualifier::Q_S32, &u32);
        REQUIRE(u32 == 0x12345678);
    }
    
    SECTION("Int16 parsing - positive value") {
        parseImmediate("32767", Qualifier::Q_S16, &u16);
        REQUIRE(u16 == 32767);
    }
    
    SECTION("Int16 parsing - negative value") {
        parseImmediate("-16384", Qualifier::Q_S16, &u16);
        REQUIRE(u16 == static_cast<uint16_t>(-16384));
    }
    
    SECTION("Int16 parsing - hex value") {
        parseImmediate("0xABCD", Qualifier::Q_U16, &u16);
        REQUIRE(u16 == 0xABCD);
    }
    
    SECTION("Int8 parsing - positive value") {
        parseImmediate("127", Qualifier::Q_S8, &u8);
        REQUIRE(u8 == 127);
    }
    
    SECTION("Int8 parsing - negative value") {
        parseImmediate("-64", Qualifier::Q_S8, &u8);
        REQUIRE(u8 == static_cast<uint8_t>(-64));
    }
    
    SECTION("Int8 parsing - hex value") {
        parseImmediate("0xFF", Qualifier::Q_U8, &u8);
        REQUIRE(u8 == 0xFF);
    }
    
    SECTION("Predicate parsing - zero should be false") {
        parseImmediate("0", Qualifier::Q_PRED, &pred);
        REQUIRE(pred == 0);
    }
    
    SECTION("Predicate parsing - non-zero should be true") {
        parseImmediate("2", Qualifier::Q_PRED, &pred);
        REQUIRE(pred == 1);
    }
    
    SECTION("Predicate parsing - negative should be true") {
        parseImmediate("-5", Qualifier::Q_PRED, &pred);
        REQUIRE(pred == 1);
    }
    
    SECTION("Predicate parsing - large value should be true") {
        parseImmediate("100", Qualifier::Q_PRED, &pred);
        REQUIRE(pred == 1);
    }
    
    SECTION("Invalid qualifier handling") {
        char buffer[8];
        memset(buffer, 0, 8);
        parseImmediate("invalid", Qualifier::S_UNKNOWN, buffer);
        // Should zero the output buffer
        bool all_zero = true;
        for (int i = 0; i < 8; i++) {
            if (buffer[i] != 0) {
                all_zero = false;
                break;
            }
        }
        REQUIRE(all_zero);
    }
    
    SECTION("Empty string handling") {
        memset(&f32, 0xFF, sizeof(f32)); // Fill with non-zero values
        parseImmediate("", Qualifier::Q_F32, &f32);
        REQUIRE(f32 == 0.0f);
    }
    
    SECTION("Whitespace handling") {
        parseImmediate("  123  ", Qualifier::Q_U32, &u32);
        REQUIRE(u32 == 123);
    }
    
    SECTION("IEEE 754 special values - infinity") {
        parseImmediate("inf", Qualifier::Q_F32, &f32);
        REQUIRE(std::isinf(f32));
    }
    
    SECTION("IEEE 754 special values - negative infinity") {
        parseImmediate("-inf", Qualifier::Q_F32, &f32);
        REQUIRE((std::isinf(f32) && f32 < 0));
    }
    
    SECTION("IEEE 754 special values - NaN") {
        parseImmediate("nan", Qualifier::Q_F32, &f32);
        REQUIRE(std::isnan(f32));
    }
    
    SECTION("Large numbers") {
        parseImmediate("9223372036854775807", Qualifier::Q_S64, &u64); // Max int64
        REQUIRE(u64 == 9223372036854775807ULL);
    }
    
    SECTION("Scientific notation") {
        parseImmediate("1.23e2", Qualifier::Q_F32, &f32);
        REQUIRE(f32 == 123.0f);
    }
}