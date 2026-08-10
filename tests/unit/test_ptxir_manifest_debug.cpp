#include <catch_amalgamated.hpp>
#include <cstdio>
#include <fstream>
#include <vector>
#include "cudart/ptxir_loader.h"
#include "ptx_ir/ptxir_format.h"

TEST_CASE("Debug: read manifest from PTXIR fixture", "[unit][ptxir][debug]") {
    std::ifstream f("tests/ptxir/fixtures/cute_rmsnorm.ptxir", std::ios::binary);
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
