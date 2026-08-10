#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>
#include <catch_amalgamated.hpp>
#include "cudart/cpptlm_module.h"

namespace fs = std::filesystem;

static fs::path fixture_path() {
    const char* env = std::getenv("PTXIR_FIXTURE_DIR");
    if (env && fs::exists(fs::path(env) / "cute_rmsnorm.ptxir")) {
        return fs::path(env) / "cute_rmsnorm.ptxir";
    }
    fs::path p = fs::current_path();
    while (!p.empty()) {
        if (fs::exists(p / "tests" / "ptxir" / "fixtures" / "cute_rmsnorm.ptxir")) {
            return p / "tests" / "ptxir" / "fixtures" / "cute_rmsnorm.ptxir";
        }
        if (p == p.parent_path()) break;
        p = p.parent_path();
    }
    return fs::path("tests/ptxir/fixtures/cute_rmsnorm.ptxir");
}

static std::vector<uint8_t> read_fixture() {
    fs::path p = fixture_path();
    REQUIRE(fs::exists(p));
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

TEST_CASE("ptxemu_module_version: returns CPPTLM_MODULE_VERSION", "[unit][cpptlm_module]") {
    REQUIRE(ptxemu_module_version() == CPPTLM_MODULE_VERSION);
    REQUIRE(CPPTLM_MODULE_VERSION == 1);
}

TEST_CASE("ptxemu_image_load: standalone PTXIR returns valid handle", "[unit][cpptlm_module]") {
    auto bytes = read_fixture();
    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);
    REQUIRE(ptxemu_image_unload(handle) == 0);
}

TEST_CASE("ptxemu_image_load: zero size returns 0", "[unit][cpptlm_module]") {
    REQUIRE(ptxemu_image_load(nullptr, 0) == 0);
}

TEST_CASE("ptxemu_image_load: corrupt magic returns 0", "[unit][cpptlm_module]") {
    std::vector<uint8_t> bad = {'X','X','X','X', 0,0,0,0, 0,0,0,0};
    REQUIRE(ptxemu_image_load(bad.data(), bad.size()) == 0);
}

TEST_CASE("ptxemu_image_kernel_name: valid handle returns kernel string", "[unit][cpptlm_module]") {
    auto bytes = read_fixture();
    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);
    char buf[256] = {0};
    int rc = ptxemu_image_kernel_name(handle, buf, sizeof(buf));
    REQUIRE(rc == 0);
    REQUIRE(std::string(buf) == "_Z14rmsnorm_kernelIfEvPKT_PS0_iif");
    REQUIRE(ptxemu_image_unload(handle) == 0);
}

TEST_CASE("ptxemu_image_kernel_name: invalid handle returns -EINVAL", "[unit][cpptlm_module]") {
    char buf[256] = {0};
    REQUIRE(ptxemu_image_kernel_name(0, buf, sizeof(buf)) == -EINVAL);
    REQUIRE(ptxemu_image_kernel_name(0xDEADBEEF, buf, sizeof(buf)) == -EINVAL);
}

TEST_CASE("ptxemu_image_execute: valid handle returns 0", "[unit][cpptlm_module]") {
    auto bytes = read_fixture();
    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);
    void* args[] = {nullptr};
    int rc = ptxemu_image_execute(handle, 1, 1, 1, 32, 1, 1, 0, args, 0);
    REQUIRE(rc == 0);
    REQUIRE(ptxemu_image_unload(handle) == 0);
}

TEST_CASE("ptxemu_image_execute: zero handle returns -EINVAL", "[unit][cpptlm_module]") {
    void* args[] = {nullptr};
    REQUIRE(ptxemu_image_execute(0, 1, 1, 1, 32, 1, 1, 0, args, 0) == -EINVAL);
}

TEST_CASE("ptxemu_image_execute: unknown handle returns -EINVAL", "[unit][cpptlm_module]") {
    void* args[] = {nullptr};
    REQUIRE(ptxemu_image_execute(0xDEADBEEF, 1, 1, 1, 32, 1, 1, 0, args, 0) == -EINVAL);
}

TEST_CASE("ptxemu_image_unload: 0 handle returns -EINVAL", "[unit][cpptlm_module]") {
    REQUIRE(ptxemu_image_unload(0) == -EINVAL);
}

TEST_CASE("ptxemu_image_unload: already-unloaded handle returns -EINVAL", "[unit][cpptlm_module]") {
    auto bytes = read_fixture();
    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);
    REQUIRE(ptxemu_image_unload(handle) == 0);
    REQUIRE(ptxemu_image_unload(handle) == -EINVAL);
}
