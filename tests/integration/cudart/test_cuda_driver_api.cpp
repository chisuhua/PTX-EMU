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
using ptxemu::cudart::CUDA_ERROR_NOT_FOUND;

namespace {

// Build minimal PTXIR image: PTXIR header + serialized statements
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

std::vector<uint8_t> build_multi_kernel_ptxir() {
    ptxemu::ir::StatementContext stmt;
    stmt.type = S_LABEL;
    stmt.data = LabelInstr{"L0"};

    std::ostringstream oss(std::ios::binary);
    PtxirWriter writer(oss);
    ManifestSection manifest;
    manifest.ptx_address_size = 64;
    manifest.kernel_name = "vec_add";
    manifest.kernels = {
        {"vec_add", 0, 0},
        {"mat_mul", 0, 0},
        {"reduce_sum", 0, 0},
    };
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

std::vector<uint8_t> build_duplicate_kernel_ptxir() {
    ptxemu::ir::StatementContext stmt;
    stmt.type = S_LABEL;
    stmt.data = LabelInstr{"L0"};

    std::ostringstream oss(std::ios::binary);
    PtxirWriter writer(oss);
    ManifestSection manifest;
    manifest.ptx_address_size = 64;
    manifest.kernel_name = "duplicate";
    manifest.kernels = {
        {"duplicate", 0, 0},
        {"duplicate", 0, 0},
        {"unique", 0, 0},
    };
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

    auto& reg = global_registry();
    REQUIRE(reg.lookup(mod) != nullptr);
    REQUIRE(reg.lookup_function(fn) != nullptr);

    REQUIRE(cuModuleUnload(mod) == CUDA_SUCCESS);

    REQUIRE(reg.lookup(mod) == nullptr);
    REQUIRE(reg.lookup_function(fn) == nullptr);
}

TEST_CASE("E2E: cuModuleGetFunction returns CUDA_ERROR_NOT_FOUND for unknown kernel",
          "[integration][cudart][driver-e2e]") {
    auto image = build_minimal_ptxir();
    CUmodule mod = nullptr;
    REQUIRE(cuModuleLoadData(&mod, image.data()) == CUDA_SUCCESS);

    CUfunction fn = nullptr;
    REQUIRE(cuModuleGetFunction(&fn, mod, "nonexistent_kernel") == CUDA_ERROR_NOT_FOUND);
    REQUIRE(fn == nullptr);

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
    REQUIRE(reg.lookup(mod1) != reg.lookup(mod2));

    REQUIRE(cuModuleUnload(mod1) == CUDA_SUCCESS);
    REQUIRE(reg.lookup(mod1) == nullptr);
    REQUIRE(reg.lookup(mod2) != nullptr);

    REQUIRE(cuModuleUnload(mod2) == CUDA_SUCCESS);
}

// ============================================================================

TEST_CASE("C3: multi-kernel: three distinct kernels return three distinct handles",
          "[integration][cudart][driver-multi-kernel]") {
    auto image = build_multi_kernel_ptxir();
    CUmodule mod = nullptr;
    REQUIRE(cuModuleLoadData(&mod, image.data()) == CUDA_SUCCESS);

    CUfunction fn_vec_add = nullptr, fn_mat_mul = nullptr, fn_reduce_sum = nullptr;
    REQUIRE(cuModuleGetFunction(&fn_vec_add, mod, "vec_add") == CUDA_SUCCESS);
    REQUIRE(cuModuleGetFunction(&fn_mat_mul, mod, "mat_mul") == CUDA_SUCCESS);
    REQUIRE(cuModuleGetFunction(&fn_reduce_sum, mod, "reduce_sum") == CUDA_SUCCESS);

    REQUIRE(fn_vec_add != nullptr);
    REQUIRE(fn_mat_mul != nullptr);
    REQUIRE(fn_reduce_sum != nullptr);
    REQUIRE(fn_vec_add != fn_mat_mul);
    REQUIRE(fn_mat_mul != fn_reduce_sum);
    REQUIRE(fn_vec_add != fn_reduce_sum);

    auto& reg = global_registry();
    auto* rec = reg.lookup(mod);
    REQUIRE(rec != nullptr);
    REQUIRE(rec->name_to_function.size() == 3);

    REQUIRE(cuModuleUnload(mod) == CUDA_SUCCESS);
}

TEST_CASE("C3: multi-kernel: duplicate name → first-match wins (SC-8)",
          "[integration][cudart][driver-multi-kernel]") {
    auto image = build_duplicate_kernel_ptxir();
    CUmodule mod = nullptr;
    REQUIRE(cuModuleLoadData(&mod, image.data()) == CUDA_SUCCESS);

    CUfunction fn_first = nullptr, fn_second = nullptr;
    REQUIRE(cuModuleGetFunction(&fn_first, mod, "duplicate") == CUDA_SUCCESS);
    REQUIRE(cuModuleGetFunction(&fn_second, mod, "duplicate") == CUDA_SUCCESS);

    REQUIRE(fn_first != nullptr);
    REQUIRE(fn_second != nullptr);
    REQUIRE(fn_first == fn_second);

    auto& reg = global_registry();
    auto* rec = reg.lookup(mod);
    REQUIRE(rec != nullptr);
    REQUIRE(rec->name_to_function["duplicate"] == fn_first);

    REQUIRE(cuModuleUnload(mod) == CUDA_SUCCESS);
}

TEST_CASE("C3: multi-kernel: not-found name returns CUDA_ERROR_NOT_FOUND",
          "[integration][cudart][driver-multi-kernel]") {
    auto image = build_multi_kernel_ptxir();
    CUmodule mod = nullptr;
    REQUIRE(cuModuleLoadData(&mod, image.data()) == CUDA_SUCCESS);

    CUfunction fn = nullptr;
    REQUIRE(cuModuleGetFunction(&fn, mod, "nonexistent_kernel") == CUDA_ERROR_NOT_FOUND);
    REQUIRE(fn == nullptr);

    REQUIRE(cuModuleUnload(mod) == CUDA_SUCCESS);
}
