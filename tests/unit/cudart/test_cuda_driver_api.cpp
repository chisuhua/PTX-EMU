// test_cuda_driver_api.cpp
// =============================================================================
// Unit tests: CUDA Driver API contract for cuModuleLoadData (Task 2.3)
//
// RED PHASE: cuModuleLoadData / cuModuleUnload symbols are not yet defined
// in cudart_sim.cpp (implementation comes in Task 2.4).
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/cudart_intrinsics.h"  // CUresult / CUmodule / CUfunction
#include "cudart/module_registry.h"
#include "ptx_ir/ptxir_format.h"    // ManifestSection
#include "ptx_ir/ptxir_reader.h"       // PtxirReader
#include "ptx_ir/ptxir_writer.h"      // PtxirWriter
#include <cstring>
#include <sstream>
#include <vector>

// Declare Driver API entry points directly (no header — implemented in Task 2.4).
// Using extern "C" avoids name mangling; the linker resolves from cudart_sim.cpp.
extern "C" {
    CUresult cuModuleLoadData(CUmodule* module, const void* image);
    CUresult cuModuleUnload(CUmodule module);
}

namespace {

// Build a minimal valid PTXIR binary section: "PTXIR" + size + serialized body.
// Uses PtxirWriter to produce a minimal statement sequence.
std::vector<uint8_t> build_minimal_ptxir() {
    ptxemu::ir::StatementContext stmt;
    stmt.type = S_LABEL;
    stmt.data = LabelInstr{"L0"};

    std::ostringstream oss(std::ios::binary);
    PtxirWriter writer(oss);
    ManifestSection manifest;
    manifest.kernel_name = "k";
    manifest.ptx_address_size = 64;
    writer.set_manifest(manifest);
    writer.write({stmt});

    std::string s = oss.str();
    std::vector<uint8_t> body(s.begin(), s.end());

    std::vector<uint8_t> image;
    image.insert(image.end(), {'P','T','X','I','R'});  // magic
    uint32_t size_le = static_cast<uint32_t>(body.size());
    image.push_back(static_cast<uint8_t>(size_le & 0xFF));
    image.push_back(static_cast<uint8_t>((size_le >> 8) & 0xFF));
    image.push_back(static_cast<uint8_t>((size_le >> 16) & 0xFF));
    image.push_back(static_cast<uint8_t>((size_le >> 24) & 0xFF));
    image.insert(image.end(), body.begin(), body.end());
    return image;
}

}  // anonymous namespace

TEST_CASE("DriverAPI: cuModuleLoadData accepts PTXIR → handle", "[unit][cudart][driver]") {
    auto image = build_minimal_ptxir();
    CUmodule mod = nullptr;
    CUresult res = cuModuleLoadData(&mod, image.data());
    REQUIRE(res == CUDA_SUCCESS);
    REQUIRE(mod != nullptr);
    REQUIRE(cuModuleUnload(mod) == CUDA_SUCCESS);
}

TEST_CASE("DriverAPI: cuModuleLoadData rejects cubin → INVALID_IMAGE", "[unit][cudart][driver]") {
    std::vector<uint8_t> bytes = {0x7f, 'E','L','F', 0,0,0,0};
    CUmodule mod = nullptr;
    CUresult res = cuModuleLoadData(&mod, bytes.data());
    REQUIRE(res == ptxemu::cudart::CUDA_ERROR_INVALID_IMAGE);
}

TEST_CASE("DriverAPI: cuModuleLoadData rejects fatbin → INVALID_IMAGE", "[unit][cudart][driver]") {
    std::vector<uint8_t> bytes = {0xBA, 0x55, 0xED, 0x10};
    CUmodule mod = nullptr;
    CUresult res = cuModuleLoadData(&mod, bytes.data());
    REQUIRE(res == ptxemu::cudart::CUDA_ERROR_INVALID_IMAGE);
}
