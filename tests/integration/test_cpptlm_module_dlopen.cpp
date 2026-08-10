#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>
#include <dlfcn.h>
#include <catch_amalgamated.hpp>
#include "cudart/cpptlm_module.h"

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

static fs::path lib_path() {
    fs::path p = fs::current_path();
    while (!p.empty()) {
        if (fs::exists(p / "lib" / "libptxemu_device.so")) {
            return p / "lib" / "libptxemu_device.so";
        }
        if (p == p.parent_path()) break;
        p = p.parent_path();
    }
    return fs::path("lib/libptxemu_device.so");
}

TEST_CASE("DL-isolated: dlopen libptxemu_device.so without libcudart.so dependency", "[integration][cpptlm_module][dlopen]") {
    fs::path lib = lib_path();
    REQUIRE(fs::exists(lib));

    void* handle = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    REQUIRE(handle != nullptr);

    using load_fn = uint64_t(*)(const uint8_t*, size_t);
    using execute_fn = int(*)(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, void**, size_t);
    using unload_fn = int(*)(uint64_t);
    using kn_fn = int(*)(uint64_t, char*, size_t);
    using ver_fn = int(*)(void);

    auto sym_load = (load_fn)dlsym(handle, "ptxemu_image_load");
    auto sym_kernel = (kn_fn)dlsym(handle, "ptxemu_image_kernel_name");
    auto sym_execute = (execute_fn)dlsym(handle, "ptxemu_image_execute");
    auto sym_unload = (unload_fn)dlsym(handle, "ptxemu_image_unload");
    auto sym_version = (ver_fn)dlsym(handle, "ptxemu_module_version");

    REQUIRE(sym_load != nullptr);
    REQUIRE(sym_kernel != nullptr);
    REQUIRE(sym_execute != nullptr);
    REQUIRE(sym_unload != nullptr);
    REQUIRE(sym_version != nullptr);
    REQUIRE(sym_version() == CPPTLM_MODULE_VERSION);

    auto p = fixture_path();
    REQUIRE(fs::exists(p));
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> bytes(sz);
    f.read(reinterpret_cast<char*>(bytes.data()), sz);

    uint64_t h = sym_load(bytes.data(), bytes.size());
    REQUIRE(h != 0);

    void* args[] = {nullptr};
    int rc = sym_execute(h, 1, 1, 1, 32, 1, 1, 0, args, 0);
    REQUIRE(rc == 0);

    REQUIRE(sym_unload(h) == 0);

    dlclose(handle);
}
