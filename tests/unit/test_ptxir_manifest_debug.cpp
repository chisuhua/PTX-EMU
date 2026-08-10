#include <catch_amalgamated.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>
#include "cudart/ptxir_loader.h"
#include "ptx_ir/ptxir_format.h"

namespace fs = std::filesystem;

TEST_CASE("Debug: read manifest from PTXIR fixture", "[unit][ptxir][debug]") {
    fs::path p = fs::path(std::getenv("PROJECT_SOURCE_DIR") ? std::getenv("PROJECT_SOURCE_DIR") : ".")
                 / "tests" / "ptxir" / "fixtures" / "cute_rmsnorm.ptxir";
    if (!fs::exists(p)) {
        fs::path cwd = fs::current_path();
        while (!cwd.empty()) {
            if (fs::exists(cwd / "tests" / "ptxir" / "fixtures" / "cute_rmsnorm.ptxir")) {
                p = cwd / "tests" / "ptxir" / "fixtures" / "cute_rmsnorm.ptxir";
                break;
            }
            if (cwd == cwd.parent_path()) break;
            cwd = cwd.parent_path();
        }
    }
    REQUIRE(fs::exists(p));
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    INFO("File size: " << sz);
    auto manifest = cudart::read_manifest_from_ptxir_section(buf.data(), buf.size());
    INFO("kernel_name: '" << manifest.kernel_name << "' size=" << manifest.kernel_name.size());
    INFO("ptx_address_size: " << manifest.ptx_address_size);
    INFO("params: " << manifest.params.size());
    for (auto& p : manifest.params) {
        INFO("  param: " << p.name << " size=" << p.size);
    }
    REQUIRE(true);
}
