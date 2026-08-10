#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>
#include <catch_amalgamated.hpp>
#include <openssl/sha.h>
#include <cstring>
#include "cudart/cpptlm_module.h"
#include "cudart/ptxir_loader.h"
#include "ptx_ir/statement_context.h"

namespace fs = std::filesystem;

static fs::path fixture_path() {
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

TEST_CASE("D3 (a): double-deserialize byte-identity", "[unit][cpptlm_module][mutation]") {
    auto p = fixture_path();
    REQUIRE(fs::exists(p));
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> bytes(sz);
    f.read(reinterpret_cast<char*>(bytes.data()), sz);

    auto stmts1 = cudart::PTXIRLoader::deserializeForCubin(bytes.data(), bytes.size());
    auto stmts2 = cudart::PTXIRLoader::deserializeForCubin(bytes.data(), bytes.size());

    REQUIRE(stmts1.size() == stmts2.size());
    REQUIRE(stmts1.size() > 0);
    for (size_t i = 0; i < stmts1.size(); ++i) {
        REQUIRE(stmts1[i].type == stmts2[i].type);
    }
}

TEST_CASE("D3 (b): N=100 sequential launches deterministic", "[unit][cpptlm_module][mutation]") {
    auto p = fixture_path();
    REQUIRE(fs::exists(p));
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> bytes(sz);
    f.read(reinterpret_cast<char*>(bytes.data()), sz);

    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);

    void* args[] = {nullptr};
    for (int i = 0; i < 100; ++i) {
        uint32_t bx = 32;
        REQUIRE(ptxemu_image_execute(handle, 1, 1, 1, bx, 1, 1, 0, args, 0) == 0);
    }
    ptxemu_image_unload(handle);
}

TEST_CASE("D3 (c): image bytes SHA-256 unchanged after N launches", "[unit][cpptlm_module][mutation]") {
    auto p = fixture_path();
    REQUIRE(fs::exists(p));
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> bytes(sz);
    f.read(reinterpret_cast<char*>(bytes.data()), sz);

    unsigned char hash_before[SHA256_DIGEST_LENGTH];
    SHA256(bytes.data(), bytes.size(), hash_before);

    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);

    void* args[] = {nullptr};
    for (int i = 0; i < 100; ++i) {
        REQUIRE(ptxemu_image_execute(handle, 1, 1, 1, 32, 1, 1, 0, args, 0) == 0);
    }
    ptxemu_image_unload(handle);

    unsigned char hash_after[SHA256_DIGEST_LENGTH];
    SHA256(bytes.data(), bytes.size(), hash_after);

    REQUIRE(std::memcmp(hash_before, hash_after, SHA256_DIGEST_LENGTH) == 0);
}
