#include "catch_amalgamated.hpp"
#include "ptx_float_arith.cuh"
#include <cmath>
#include <cstdint>

// Helper function to compare floats with tolerance
bool float_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) <= epsilon;
}

bool double_equal(double a, double b, double epsilon = 1e-10) {
    return std::abs(a - b) <= epsilon;
}

TEST_CASE("PTX: add.f32", "[ptx][float][add]") {
    float result;
    test_ptx_add_f32(5.5f, 7.3f, &result);
    REQUIRE(float_equal(result, 12.8f));
}

TEST_CASE("PTX: sub.f32", "[ptx][float][sub]") {
    float result;
    test_ptx_sub_f32(10.7f, 3.2f, &result);
    REQUIRE(float_equal(result, 7.5f));
}

TEST_CASE("PTX: mul.f32", "[ptx][float][mul]") {
    float result;
    test_ptx_mul_f32(6.5f, 7.2f, &result);
    REQUIRE(float_equal(result, 46.8f));
}

TEST_CASE("PTX: div.f32", "[ptx][float][div]") {
    float result;
    test_ptx_div_f32(15.0f, 3.0f, &result);
    REQUIRE(float_equal(result, 5.0f));
}

TEST_CASE("PTX: abs.f32", "[ptx][float][abs]") {
    float result;
    test_ptx_abs_f32(-42.7f, &result);
    REQUIRE(float_equal(result, 42.7f));
}

TEST_CASE("PTX: neg.f32", "[ptx][float][neg]") {
    float result;
    test_ptx_neg_f32(100.5f, &result);
    REQUIRE(float_equal(result, -100.5f));
}

TEST_CASE("PTX: min.f32", "[ptx][float][min]") {
    float result;
    test_ptx_min_f32(-5.3f, 10.8f, &result);
    REQUIRE(float_equal(result, -5.3f));
}

TEST_CASE("PTX: max.f32", "[ptx][float][max]") {
    float result;
    test_ptx_max_f32(-5.3f, 10.8f, &result);
    REQUIRE(float_equal(result, 10.8f));
}

TEST_CASE("PTX: sqrt.f32", "[ptx][float][sqrt]") {
    float result;
    test_ptx_sqrt_f32(16.0f, &result);
    REQUIRE(float_equal(result, 4.0f));
}

TEST_CASE("PTX: rcp.f32", "[ptx][float][rcp]") {
    float result;
    test_ptx_rcp_f32(4.0f, &result);
    REQUIRE(float_equal(result, 0.25f));
}

TEST_CASE("PTX: add.f64", "[ptx][double][add]") {
    double result;
    test_ptx_add_f64(5.5, 7.3, &result);
    REQUIRE(double_equal(result, 12.8));
}

TEST_CASE("PTX: sub.f64", "[ptx][double][sub]") {
    double result;
    test_ptx_sub_f64(10.7, 3.2, &result);
    REQUIRE(double_equal(result, 7.5));
}

TEST_CASE("PTX: mul.f64", "[ptx][double][mul]") {
    double result;
    test_ptx_mul_f64(6.5, 7.2, &result);
    REQUIRE(double_equal(result, 46.8));
}

TEST_CASE("PTX: div.f64", "[ptx][double][div]") {
    double result;
    test_ptx_div_f64(15.0, 3.0, &result);
    REQUIRE(double_equal(result, 5.0));
}

TEST_CASE("PTX: abs.f64", "[ptx][double][abs]") {
    double result;
    test_ptx_abs_f64(-42.7, &result);
    REQUIRE(double_equal(result, 42.7));
}

TEST_CASE("PTX: neg.f64", "[ptx][double][neg]") {
    double result;
    test_ptx_neg_f64(100.5, &result);
    REQUIRE(double_equal(result, -100.5));
}

TEST_CASE("PTX: min.f64", "[ptx][double][min]") {
    double result;
    test_ptx_min_f64(-5.3, 10.8, &result);
    REQUIRE(double_equal(result, -5.3));
}

TEST_CASE("PTX: max.f64", "[ptx][double][max]") {
    double result;
    test_ptx_max_f64(-5.3, 10.8, &result);
    REQUIRE(double_equal(result, 10.8));
}

TEST_CASE("PTX: sqrt.f64", "[ptx][double][sqrt]") {
    double result;
    test_ptx_sqrt_f64(16.0, &result);
    REQUIRE(double_equal(result, 4.0));
}

TEST_CASE("PTX: rcp.f64", "[ptx][double][rcp]") {
    double result;
    test_ptx_rcp_f64(4.0, &result);
    REQUIRE(double_equal(result, 0.25));
}