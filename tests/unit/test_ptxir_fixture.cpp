#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <catch_amalgamated.hpp>
#include "ptxir/ptxir_serialization.h"

TEST_CASE("Generate PTXIR fixture: bench/cute/cute_rmsnorm.ptx", "[unit][ptxir][fixture]") {
    const char* src = std::getenv("PTX_FIXTURE_SRC");
    std::string ptx_path = src ? src : "bench/cute/cute_rmsnorm.ptx";
    std::string out_path = "tests/ptxir/fixtures/cute_rmsnorm.ptxir";

    if (std::filesystem::exists(out_path)) {
        SUCCEED("Fixture already exists");
        return;
    }
    std::filesystem::create_directories(std::filesystem::path(out_path).parent_path());
    REQUIRE(generate_ptxir(ptx_path, out_path, "_Z14rmsnorm_kernelIfEvPKT_PS0_iif"));
    REQUIRE(std::filesystem::exists(out_path));
    REQUIRE(std::filesystem::file_size(out_path) > 0);
}
