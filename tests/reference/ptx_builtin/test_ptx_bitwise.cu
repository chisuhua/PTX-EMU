#include "catch_amalgamated.hpp"
#include "ptx_bitwise_shift.cuh"
#include <cstdint>

TEST_CASE("PTX: and.b32", "[ptx][bitwise][and][b32]") {
    uint32_t result;
    test_ptx_and_b32(0xF0F0F0F0, 0x0F0F0F0F, &result);
    REQUIRE(result == 0x00000000);
}

TEST_CASE("PTX: or.b32", "[ptx][bitwise][or][b32]") {
    uint32_t result;
    test_ptx_or_b32(0xF0F0F0F0, 0x0F0F0F0F, &result);
    REQUIRE(result == 0xFFFFFFFF);
}

TEST_CASE("PTX: xor.b32", "[ptx][bitwise][xor][b32]") {
    uint32_t result;
    test_ptx_xor_b32(0xF0F0F0F0, 0x0F0F0F0F, &result);
    REQUIRE(result == 0xFFFFFFFF);
}

TEST_CASE("PTX: not.b32", "[ptx][bitwise][not][b32]") {
    uint32_t result;
    test_ptx_not_b32(0xAAAAAAAA, &result);
    REQUIRE(result == 0x55555555);
}

TEST_CASE("PTX: shl.b32", "[ptx][bitwise][shl][b32]") {
    uint32_t result;
    test_ptx_shl_b32(0x00000001, 4, &result);
    REQUIRE(result == 0x00000010);
}

TEST_CASE("PTX: shr.b32", "[ptx][bitwise][shr][b32]") {
    uint32_t result;
    test_ptx_shr_b32(0x10000000, 4, &result);
    REQUIRE(result == 0x01000000);
}

TEST_CASE("PTX: and.b64", "[ptx][bitwise][and][b64]") {
    uint64_t result;
    test_ptx_and_b64(0xF0F0F0F0F0F0F0F0ULL, 0x0F0F0F0F0F0F0F0FULL, &result);
    REQUIRE(result == 0x0000000000000000ULL);
}

TEST_CASE("PTX: or.b64", "[ptx][bitwise][or][b64]") {
    uint64_t result;
    test_ptx_or_b64(0xF0F0F0F0F0F0F0F0ULL, 0x0F0F0F0F0F0F0F0FULL, &result);
    REQUIRE(result == 0xFFFFFFFFFFFFFFFFULL);
}

TEST_CASE("PTX: xor.b64", "[ptx][bitwise][xor][b64]") {
    uint64_t result;
    test_ptx_xor_b64(0xF0F0F0F0F0F0F0F0ULL, 0x0F0F0F0F0F0F0F0FULL, &result);
    REQUIRE(result == 0xFFFFFFFFFFFFFFFFULL);
}

TEST_CASE("PTX: not.b64", "[ptx][bitwise][not][b64]") {
    uint64_t result;
    test_ptx_not_b64(0xAAAAAAAAAAAAAAAAULL, &result);
    REQUIRE(result == 0x5555555555555555ULL);
}

TEST_CASE("PTX: shl.b64", "[ptx][bitwise][shl][b64]") {
    uint64_t result;
    test_ptx_shl_b64(0x0000000000000001ULL, 8, &result);
    REQUIRE(result == 0x0000000000000100ULL);
}

TEST_CASE("PTX: shr.b64", "[ptx][bitwise][shr][b64]") {
    uint64_t result;
    test_ptx_shr_b64(0x1000000000000000ULL, 8, &result);
    REQUIRE(result == 0x0010000000000000ULL);
}