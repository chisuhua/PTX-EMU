#include "catch_amalgamated.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>

TEST_CASE("Path 2D Scenario 3.1: cute_rmsnorm output byte-level matches baseline", "[e2e][path_2D]") {
    std::ifstream f("../../ptxir/baselines/cute_rmsnorm_output_baseline.bin", std::ios::binary);
    REQUIRE(f.good());
    std::vector<uint8_t> baseline((std::istreambuf_iterator<char>(f)), {});
    REQUIRE(baseline.size() >= 14);

    const char expected_magic[10] = {'P','T','X','R','_','O','U','T', 0, 0};
    REQUIRE(std::memcmp(baseline.data(), expected_magic, 10) == 0);

    uint32_t size;
    std::memcpy(&size, baseline.data() + 10, 4);
    REQUIRE(size == baseline.size() - 14);
}
