#include "catch_amalgamated.hpp"
#include "cudart/module_registry.h"
#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/ptxir_writer.h"
#include "ptx_ir/statement_context.h"
#include <sstream>
#include <vector>

// PTXIR header format: "PTXIR" (5 bytes) + body_size (4 bytes LE) + body
static constexpr size_t PTXIR_HEADER_SIZE = 9;

extern "C" {
    CUresult cuModuleLoadData(CUmodule* module, const void* image);
    CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
    CUresult cuModuleUnload(CUmodule module);
}

using ptxemu::cudart::global_registry;

namespace {

// Build minimal PTXIR image: PTXIR header + serialized statements
std::vector<uint8_t> build_minimal_ptxir() {
    StatementContext stmt;
    stmt.type = S_LABEL;
    stmt.data = LabelInstr{"L0"};

    std::ostringstream oss(std::ios::binary);
    PtxirWriter writer(oss);
    ManifestSection manifest;
    manifest.kernel_name = "k";
    manifest.ptx_address_size = 64;
    writer.set_manifest(manifest);
    writer.write({stmt});

    std::string body = oss.str();
    std::vector<uint8_t> image;
    image.insert(image.end(), {'P','T','X','I','R'});
    uint32_t sz = static_cast<uint32_t>(body.size());
    image.push_back(static_cast<uint8_t>(sz & 0xFF));
    image.push_back(static_cast<uint8_t>((sz >> 8) & 0xFF));
    image.push_back(static_cast<uint8_t>((sz >> 16) & 0xFF));
    image.push_back(static_cast<uint8_t>((sz >> 24) & 0xFF));
    image.insert(image.end(), body.begin(), body.end());
    return image;
}

}  // anonymous namespace

// ============================================================================
// Driver API E2E lifecycle tests
// ============================================================================

TEST_CASE("E2E: cuModuleLoadData → cuModuleGetFunction → cuModuleUnload lifecycle",
          "[integration][cudart][driver-e2e]") {
    auto image = build_minimal_ptxir();

    CUmodule mod = nullptr;
    REQUIRE(cuModuleLoadData(&mod, image.data()) == CUDA_SUCCESS);
    REQUIRE(mod != nullptr);

    CUfunction fn = nullptr;
    REQUIRE(cuModuleGetFunction(&fn, mod, "k") == CUDA_SUCCESS);
    REQUIRE(fn != nullptr);

    // Lookup via registry must succeed
    auto& reg = global_registry();
    REQUIRE(reg.lookup(mod) != nullptr);
    REQUIRE(reg.lookup_function(fn) != nullptr);

    // Unload
    REQUIRE(cuModuleUnload(mod) == CUDA_SUCCESS);

    // After unload: both module and function must be invalid
    REQUIRE(reg.lookup(mod) == nullptr);
    REQUIRE(reg.lookup_function(fn) == nullptr);
}

TEST_CASE("E2E: cuModuleGetFunction returns handle for any name (kernel-not-found at launch time)",
          "[integration][cudart][driver-e2e]") {
    auto image = build_minimal_ptxir();
    CUmodule mod = nullptr;
    REQUIRE(cuModuleLoadData(&mod, image.data()) == CUDA_SUCCESS);

    // Unknown kernel name still gets a CUfunction handle (handle is created).
    // Actual kernel-not-found fires at cuLaunchKernel time, not here.
    CUfunction fn = nullptr;
    REQUIRE(cuModuleGetFunction(&fn, mod, "nonexistent_kernel") == CUDA_SUCCESS);

    REQUIRE(cuModuleUnload(mod) == CUDA_SUCCESS);
}

TEST_CASE("E2E: two modules loaded independently coexist",
          "[integration][cudart][driver-e2e]") {
    auto image = build_minimal_ptxir();

    CUmodule mod1 = nullptr, mod2 = nullptr;
    REQUIRE(cuModuleLoadData(&mod1, image.data()) == CUDA_SUCCESS);
    REQUIRE(cuModuleLoadData(&mod2, image.data()) == CUDA_SUCCESS);
    REQUIRE(mod1 != mod2);

    auto& reg = global_registry();
    REQUIRE(reg.lookup(mod1) != nullptr);
    REQUIRE(reg.lookup(mod2) != nullptr);
    REQUIRE(reg.lookup(mod1) != reg.lookup(mod2));  // distinct records

    REQUIRE(cuModuleUnload(mod1) == CUDA_SUCCESS);
    REQUIRE(reg.lookup(mod1) == nullptr);
    REQUIRE(reg.lookup(mod2) != nullptr);  // mod2 unaffected

    REQUIRE(cuModuleUnload(mod2) == CUDA_SUCCESS);
}
