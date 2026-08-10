#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <catch_amalgamated.hpp>
#include "ptxir/ptxir_serialization.h"

namespace fs = std::filesystem;

static fs::path project_root() {
    const char* env = std::getenv("PROJECT_SOURCE_DIR");
    if (env) return fs::path(env);
    fs::path p = fs::current_path();
    while (!p.empty()) {
        if (fs::exists(p / "tests" / "ptxir" / "fixtures")) return p;
        if (p == p.parent_path()) break;
        p = p.parent_path();
    }
    return fs::current_path();
}

TEST_CASE("Generate PTXIR fixture: bench/cute/cute_rmsnorm.ptx", "[unit][ptxir][fixture]") {
    fs::path root = project_root();
    fs::path src = fs::path(std::getenv("PTX_FIXTURE_SRC") ? std::getenv("PTX_FIXTURE_SRC") : (root / "bench/cute/cute_rmsnorm.ptx").string());
    fs::path out_path = root / "tests" / "ptxir" / "fixtures" / "cute_rmsnorm.ptxir";

    if (fs::exists(out_path)) {
        SUCCEED("Fixture already exists");
        return;
    }
    fs::create_directories(out_path.parent_path());
    REQUIRE(generate_ptxir(src.string(), out_path.string(), "_Z14rmsnorm_kernelIfEvPKT_PS0_iif"));
    REQUIRE(fs::exists(out_path));
    REQUIRE(fs::file_size(out_path) > 0);
}
