#include "catch_amalgamated.hpp"
#include "ptx_ld_st.cuh"
#include <cmath>
#include <cstdint>

// Helper function to compare floats with tolerance
bool float_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) <= epsilon;
}

bool double_equal(double a, double b, double epsilon = 1e-10) {
    return std::abs(a - b) <= epsilon;
}

// Kernel definitions have been removed as they are now declared in the header and defined elsewhere.
// This prevents multiple definition errors during linking.

// Tests for LD (load) operations
TEST_CASE("PTX: ld.global.u8", "[ptx][ld][global][u8]") {
    uint8_t value = 42;
    uint8_t result = 0;

    uint8_t* d_data;
    uint8_t* d_result;
    cudaMalloc(&d_data, sizeof(uint8_t));
    cudaMalloc(&d_result, sizeof(uint8_t));
    
    cudaMemcpy(d_data, &value, sizeof(uint8_t), cudaMemcpyHostToDevice);

    ld_u8_kernel<<<1, 1>>>(d_data, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    REQUIRE(result == value);
    
    cudaFree(d_data);
    cudaFree(d_result);
}

TEST_CASE("PTX: ld.global.u16", "[ptx][ld][global][u16]") {
    uint16_t value = 300;
    uint16_t result = 0;

    uint16_t* d_data;
    uint16_t* d_result;
    cudaMalloc(&d_data, sizeof(uint16_t));
    cudaMalloc(&d_result, sizeof(uint16_t));
    cudaMemcpy(d_data, &value, sizeof(uint16_t), cudaMemcpyHostToDevice);

    ld_u16_kernel<<<1, 1>>>(d_data, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost);
    REQUIRE(result == value);
    
    cudaFree(d_data);
    cudaFree(d_result);
}

TEST_CASE("PTX: ld.global.u32", "[ptx][ld][global][u32]") {
    uint32_t value = 50000;
    uint32_t result = 0;

    uint32_t* d_data;
    uint32_t* d_result;
    cudaMalloc(&d_data, sizeof(uint32_t));
    cudaMalloc(&d_result, sizeof(uint32_t));
    cudaMemcpy(d_data, &value, sizeof(uint32_t), cudaMemcpyHostToDevice);

    ld_u32_kernel<<<1, 1>>>(d_data, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    REQUIRE(result == value);
    
    cudaFree(d_data);
    cudaFree(d_result);
}

TEST_CASE("PTX: ld.global.u64", "[ptx][ld][global][u64]") {
    uint64_t value = 123456789ULL;
    uint64_t result = 0;

    uint64_t* d_data;
    uint64_t* d_result;
    cudaMalloc(&d_data, sizeof(uint64_t));
    cudaMalloc(&d_result, sizeof(uint64_t));
    cudaMemcpy(d_data, &value, sizeof(uint64_t), cudaMemcpyHostToDevice);

    ld_u64_kernel<<<1, 1>>>(d_data, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    REQUIRE(result == value);
    
    cudaFree(d_data);
    cudaFree(d_result);
}

TEST_CASE("PTX: ld.global.f32", "[ptx][ld][global][f32]") {
    float value = 3.14159f;
    float result = 0.0f;

    float* d_data;
    float* d_result;
    cudaMalloc(&d_data, sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_data, &value, sizeof(float), cudaMemcpyHostToDevice);

    ld_f32_kernel<<<1, 1>>>(d_data, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    REQUIRE(float_equal(result, value));
    
    cudaFree(d_data);
    cudaFree(d_result);
}

TEST_CASE("PTX: ld.global.f64", "[ptx][ld][global][f64]") {
    double value = 2.718281828;
    double result = 0.0;

    double* d_data;
    double* d_result;
    cudaMalloc(&d_data, sizeof(double));
    cudaMalloc(&d_result, sizeof(double));
    cudaMemcpy(d_data, &value, sizeof(double), cudaMemcpyHostToDevice);

    ld_f64_kernel<<<1, 1>>>(d_data, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    REQUIRE(double_equal(result, value));
    
    cudaFree(d_data);
    cudaFree(d_result);
}

// Tests for ST (store) operations
TEST_CASE("PTX: st.global.u8", "[ptx][st][global][u8]") {
    uint8_t value = 128;
    uint8_t result = 0;

    uint8_t* d_result;
    cudaMalloc(&d_result, sizeof(uint8_t));

    st_u8_kernel<<<1, 1>>>(d_result, value);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    REQUIRE(result == value);
    cudaFree(d_result);
}

TEST_CASE("PTX: st.global.u16", "[ptx][st][global][u16]") {
    uint16_t value = 1000;
    uint16_t result = 0;

    uint16_t* d_result;
    cudaMalloc(&d_result, sizeof(uint16_t));

    st_u16_kernel<<<1, 1>>>(d_result, value);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost);
    REQUIRE(result == value);
    cudaFree(d_result);
}

TEST_CASE("PTX: st.global.u32", "[ptx][st][global][u32]") {
    uint32_t value = 100000;
    uint32_t result = 0;

    uint32_t* d_result;
    cudaMalloc(&d_result, sizeof(uint32_t));

    st_u32_kernel<<<1, 1>>>(d_result, value);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    REQUIRE(result == value);
    cudaFree(d_result);
}

TEST_CASE("PTX: st.global.u64", "[ptx][st][global][u64]") {
    uint64_t value = 9876543210ULL;
    uint64_t result = 0;

    uint64_t* d_result;
    cudaMalloc(&d_result, sizeof(uint64_t));

    st_u64_kernel<<<1, 1>>>(d_result, value);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    REQUIRE(result == value);
    cudaFree(d_result);
}

TEST_CASE("PTX: st.global.f32", "[ptx][st][global][f32]") {
    float value = 1.41421f;  // sqrt(2)
    float result = 0.0f;

    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    st_f32_kernel<<<1, 1>>>(d_result, value);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    REQUIRE(float_equal(result, value));
    cudaFree(d_result);
}

TEST_CASE("PTX: st.global.f64", "[ptx][st][global][f64]") {
    double value = 1.732050807568877;  // sqrt(3)
    double result = 0.0;

    double* d_result;
    cudaMalloc(&d_result, sizeof(double));

    st_f64_kernel<<<1, 1>>>(d_result, value);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    REQUIRE(double_equal(result, value));
    cudaFree(d_result);
}

// Tests for shared memory operations
TEST_CASE("PTX: shared load and store", "[ptx][shared][load][store]") {
    uint32_t result = 0;
    test_ptx_shared_store_only(&result);
    REQUIRE(result == 42);
}

TEST_CASE("PTX: shared load", "[ptx][shared][load]") {
    uint32_t result = 0;
    test_ptx_shared_load_store(&result);
    REQUIRE(result == 1);  // First element of input array
}