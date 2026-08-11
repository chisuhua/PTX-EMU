#include "catch_amalgamated.hpp"
#include "cudart/module_registry.h"
#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/ptxir_writer.h"
#include "ptx_ir/statement_context.h"
#include <cstring>
#include <sstream>
#include <vector>

// PTXIR header format: "PTXIR" (5 bytes) + body_size (4 bytes LE) + body
static constexpr size_t PTXIR_HEADER_SIZE = 9;

extern "C" {
    CUresult cuModuleLoadData(CUmodule* module, const void* image);
    CUresult cuModuleUnload(CUmodule module);
}

using ptxemu::cudart::global_registry;

namespace {

// FNV-1a hash for stable non-crypto comparison
static uint64_t fnv1a_hash(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;  // FNV offset basis
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

std::string short_hash(const uint8_t* p, size_t n) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016zx", fnv1a_hash(p, n));
    return std::string(buf);
}

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
// ADR-0029 D3: per-launch fresh PtxContext invariant
//
// The D3 mutation bug: multiple lookups of the same module record could
// mutate parsed_statements or image_bytes (e.g., via cached PtxContext reuse).
//
// Fix invariant: image_bytes and parsed_statements must remain bit-identical
// across N lookups of the same handle.
// ============================================================================

TEST_CASE("D3 mutation: image bytes stable across 100 lookups",
          "[integration][cudart][mutation]") {
    auto image = build_minimal_ptxir();
    std::string h0 = short_hash(image.data(), image.size());

    CUmodule mod = nullptr;
    REQUIRE(cuModuleLoadData(&mod, image.data()) == CUDA_SUCCESS);
    REQUIRE(mod != nullptr);

    auto& reg = global_registry();
    for (int i = 0; i < 100; ++i) {
        auto* rec = reg.lookup(mod);
        REQUIRE(rec != nullptr);
        REQUIRE(short_hash(rec->image_bytes.get(), rec->image_size) == h0);
    }

    REQUIRE(cuModuleUnload(mod) == CUDA_SUCCESS);
}

TEST_CASE("D3 mutation: parsed_statements not mutated across lookups",
          "[integration][cudart][mutation]") {
    auto image = build_minimal_ptxir();

    CUmodule mod = nullptr;
    REQUIRE(cuModuleLoadData(&mod, image.data()) == CUDA_SUCCESS);

    auto& reg = global_registry();
    auto* rec1 = reg.lookup(mod);
    REQUIRE(rec1 != nullptr);
    size_t initial_size = rec1->parsed_statements.size();
    REQUIRE(initial_size >= 1);

    // 100 lookups — content must stay identical
    for (int i = 0; i < 100; ++i) {
        auto* rec2 = reg.lookup(mod);
        REQUIRE(rec2 == rec1);  // same record
        REQUIRE(rec2->parsed_statements.size() == initial_size);
        REQUIRE(rec2->parsed_statements[0].type == rec1->parsed_statements[0].type);
    }

    REQUIRE(cuModuleUnload(mod) == CUDA_SUCCESS);
}

TEST_CASE("D3 mutation: two loads produce two independent deep copies",
          "[integration][cudart][mutation]") {
    auto image = build_minimal_ptxir();

    CUmodule m1 = nullptr, m2 = nullptr;
    REQUIRE(cuModuleLoadData(&m1, image.data()) == CUDA_SUCCESS);
    REQUIRE(cuModuleLoadData(&m2, image.data()) == CUDA_SUCCESS);
    REQUIRE(m1 != m2);

    auto& reg = global_registry();
    auto* r1 = reg.lookup(m1);
    auto* r2 = reg.lookup(m2);
    REQUIRE(r1 != nullptr);
    REQUIRE(r2 != nullptr);
    REQUIRE(r1 != r2);  // distinct records

    // Mutating one's image_bytes must not affect the other
    if (r1->image_size > 0) {
        r1->image_bytes[0] ^= 0xFF;
        REQUIRE(r2->image_bytes[0] != r1->image_bytes[0]);
    }

    REQUIRE(cuModuleUnload(m1) == CUDA_SUCCESS);
    REQUIRE(cuModuleUnload(m2) == CUDA_SUCCESS);
}