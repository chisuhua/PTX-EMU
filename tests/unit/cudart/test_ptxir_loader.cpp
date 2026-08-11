#include "catch_amalgamated.hpp"
#include "cudart/ptxir_loader.h"
#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/ptxir_reader.h"
#include "ptx_ir/ptxir_writer.h"
#include <cstring>
#include <fstream>
#include <openssl/sha.h>
#include <vector>

namespace {

std::vector<uint8_t> make_embedded(const std::vector<uint8_t>& prefix,
                                   const std::vector<uint8_t>& section) {
    std::vector<uint8_t> out = prefix;
    out.insert(out.end(), section.begin(), section.end());
    uint32_t size_le = static_cast<uint32_t>(section.size());
    out.push_back(static_cast<uint8_t>(size_le & 0xFF));
    out.push_back(static_cast<uint8_t>((size_le >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((size_le >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((size_le >> 24) & 0xFF));
    out.insert(out.end(), cudart::PTXIR_EMBED_MAGIC, cudart::PTXIR_EMBED_MAGIC + 8);
    return out;
}

std::vector<uint8_t> sha256(const std::vector<uint8_t>& data) {
    std::vector<uint8_t> hash(32);
    SHA256(data.data(), data.size(), hash.data());
    return hash;
}

std::vector<uint8_t> build_minimal_ptxir_section(const std::string& kernel_name) {
    StatementContext stmt;
    stmt.type = S_LABEL;
    stmt.data = LabelInstr{"L0"};

    std::ostringstream oss(std::ios::binary);
    PtxirWriter writer(oss);
    ManifestSection manifest;
    manifest.kernel_name = kernel_name;
    manifest.ptx_address_size = 64;
    writer.set_manifest(manifest);
    writer.write({stmt});

    std::string s = oss.str();
    return std::vector<uint8_t>(s.begin(), s.end());
}

std::vector<uint8_t> build_minimal_ptxir_section_with_bad_hash() {
    StatementContext stmt;
    stmt.type = S_LABEL;
    stmt.data = LabelInstr{"L0"};

    std::ostringstream oss(std::ios::binary);
    PtxirWriter writer(oss);
    ManifestSection manifest;
    manifest.kernel_name = "k";
    manifest.ptx_address_size = 64;
    manifest.cubin_hash = std::vector<uint8_t>(32, 0xFF);
    writer.set_manifest(manifest);
    writer.write({stmt});

    std::string s = oss.str();
    return std::vector<uint8_t>(s.begin(), s.end());
}

std::vector<uint8_t> build_embedded_with_hash(const std::vector<uint8_t>& prefix,
                                              const std::vector<uint8_t>& section) {
    std::vector<uint8_t> out = prefix;
    out.insert(out.end(), section.begin(), section.end());
    uint32_t size_le = static_cast<uint32_t>(section.size());
    out.push_back(static_cast<uint8_t>(size_le & 0xFF));
    out.push_back(static_cast<uint8_t>((size_le >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((size_le >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((size_le >> 24) & 0xFF));
    out.insert(out.end(), cudart::PTXIR_EMBED_MAGIC, cudart::PTXIR_EMBED_MAGIC + 8);
    return out;
}

}  // namespace

TEST_CASE("hasEmbeddedPTXIR_legitimateEmbedded_returnsTrue", "[ptxir_loader]") {
    auto bin = make_embedded({0x01, 0x02}, {0xAA, 0xBB});
    REQUIRE(cudart::PTXIRLoader::hasEmbeddedPTXIR(bin.data(), bin.size()) == true);
}

TEST_CASE("hasEmbeddedPTXIR_plainCubin_returnsFalse", "[ptxir_loader]") {
    std::vector<uint8_t> plain = {0x01, 0x02, 0x03};
    REQUIRE(cudart::PTXIRLoader::hasEmbeddedPTXIR(plain.data(), plain.size()) == false);
}

TEST_CASE("hasEmbeddedPTXIR_truncatedInput_returnsFalse", "[ptxir_loader]") {
    std::vector<uint8_t> short_input = {0x01, 0x02};
    REQUIRE(cudart::PTXIRLoader::hasEmbeddedPTXIR(short_input.data(), short_input.size()) == false);
}

TEST_CASE("hasEmbeddedPTXIR_fakeMagic_returnsFalse", "[ptxir_loader]") {
    std::vector<uint8_t> bin = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                 0x00, 0x00, 0x00, 0x00, 'P', 'T', 'X', 'R'};
    REQUIRE(cudart::PTXIRLoader::hasEmbeddedPTXIR(bin.data(), bin.size()) == false);
}

TEST_CASE("hasEmbeddedPTXIR_sizeFieldOverflows_returnsFalse", "[ptxir_loader]") {
    std::vector<uint8_t> bin(12, 0);
    bin[0] = 0x01;
    uint32_t huge_size = 1000000;
    std::memcpy(bin.data() + 4, &huge_size, sizeof(huge_size));
    std::memcpy(bin.data() + 8, cudart::PTXIR_EMBED_MAGIC, 8);
    REQUIRE(cudart::PTXIRLoader::hasEmbeddedPTXIR(bin.data(), bin.size()) == false);
}

TEST_CASE("extractPTXIR_legitimateEmbedded_returnsSection", "[ptxir_loader]") {
    auto bin = make_embedded({0x01}, {0xAA, 0xBB, 0xCC});
    size_t out_size = 0;
    auto section = cudart::PTXIRLoader::extractPTXIR(bin.data(), bin.size(), &out_size);
    REQUIRE(section != nullptr);
    REQUIRE(out_size == 3);
    REQUIRE(section[0] == 0xAA);
}

TEST_CASE("extractPTXIR_plainCubin_returnsNullptr", "[ptxir_loader]") {
    std::vector<uint8_t> plain = {0x01, 0x02, 0x03};
    size_t out_size = 0;
    auto section = cudart::PTXIRLoader::extractPTXIR(plain.data(), plain.size(), &out_size);
    REQUIRE(section == nullptr);
}

TEST_CASE("extractPTXIR_zeroSizeInput_returnsNullptr", "[ptxir_loader]") {
    size_t out_size = 0;
    auto section = cudart::PTXIRLoader::extractPTXIR(nullptr, 0, &out_size);
    REQUIRE(section == nullptr);
}

TEST_CASE("extractPureCubin_legitimateEmbedded_returnsBytes", "[ptxir_loader]") {
    auto section = build_minimal_ptxir_section("k");
    std::vector<uint8_t> prefix = {0x01, 0x02, 0x03};
    auto bin = build_embedded_with_hash(prefix, section);
    auto pure = cudart::PTXIRLoader::extractPureCubin(bin.data(), bin.size());
    REQUIRE(pure.has_value());
    REQUIRE(pure->size() == 3);
    REQUIRE((*pure)[0] == 0x01);
}

TEST_CASE("extractPureCubin_plainCubin_passthrough", "[ptxir_loader]") {
    std::vector<uint8_t> plain = {0x01, 0x02, 0x03};
    auto pure = cudart::PTXIRLoader::extractPureCubin(plain.data(), plain.size());
    REQUIRE(pure.has_value());
    REQUIRE(*pure == plain);
}

TEST_CASE("extractPureCubin_hashMismatch_returnsNullopt", "[ptxir_loader]") {
    std::vector<uint8_t> prefix = {0x01};
    auto section = build_minimal_ptxir_section_with_bad_hash();
    auto bin = build_embedded_with_hash(prefix, section);
    bin[0] = 0xFF;
    auto pure = cudart::PTXIRLoader::extractPureCubin(bin.data(), bin.size());
    REQUIRE(!pure.has_value());
}

TEST_CASE("deserializeForCubin_legitimateSection_returnsContexts", "[ptxir_loader]") {
    auto section = build_minimal_ptxir_section("test_kernel");
    auto stmts = cudart::PTXIRLoader::deserializeForCubin(section.data(), section.size());
    REQUIRE(!stmts.empty());
}

TEST_CASE("deserializeForCubin_corruptedHeader_returnsEmpty", "[ptxir_loader]") {
    std::vector<uint8_t> corrupted = {0xFF, 0xFF, 0xFF, 0xFF};
    auto stmts = cudart::PTXIRLoader::deserializeForCubin(corrupted.data(), corrupted.size());
    REQUIRE(stmts.empty());
}

TEST_CASE("deserializeForCubin_hashCheckNotNeeded_returnsContexts", "[ptxir_loader]") {
    auto section = build_minimal_ptxir_section("test_kernel");
    auto stmts = cudart::PTXIRLoader::deserializeForCubin(section.data(), section.size());
    REQUIRE(!stmts.empty());
}

TEST_CASE("extractPTXIR_nullptr_returnsNullptr", "[ptxir_loader]") {
    size_t out_size = 0;
    auto section = cudart::PTXIRLoader::extractPTXIR(nullptr, 0, &out_size);
    REQUIRE(section == nullptr);
    REQUIRE(out_size == 0);
}

TEST_CASE("read_manifest_from_ptxir_section_nullptr_returnsEmpty", "[ptxir_loader]") {
    auto m = cudart::read_manifest_from_ptxir_section(nullptr, 0);
    REQUIRE(m.kernel_name.empty());
    REQUIRE(m.cubin_hash.empty());
    REQUIRE(m.params.empty());
}

TEST_CASE("read_manifest_from_ptxir_section_emptyData_returnsEmpty", "[ptxir_loader]") {
    auto m = cudart::read_manifest_from_ptxir_section(nullptr, 0);
    REQUIRE(m.kernel_name.empty());
}

TEST_CASE("extractPureCubin_nullptr_passthroughOrEmpty", "[ptxir_loader]") {
    auto pure = cudart::PTXIRLoader::extractPureCubin(nullptr, 0);
    REQUIRE(pure.has_value() == false);
}

TEST_CASE("PTXIR_VERSION bumped after ADR-0028", "[ptxir_loader][version]") {
    // PTXIR_VERSION is bumped to 4 per ADR-0028 + ADR-0023 Extend-Only.
    REQUIRE(PTXIR_VERSION >= 4);
}

TEST_CASE("KernelEntry struct accessible after ADR-0028", "[ptxir_loader][version]") {
    // ADR-0028 adds KernelEntry struct to ptxir_format.h.
    KernelEntry entry;
    entry.name = "test_kernel";
    entry.arg_count = 1;
    entry.arg_byte_size = 8;
    REQUIRE(entry.name == "test_kernel");
    REQUIRE(entry.arg_count == 1);
    REQUIRE(entry.arg_byte_size == 8);
}

TEST_CASE("ManifestSection v2 has kernels vector after ADR-0028", "[ptxir_loader][version]") {
    ManifestSection ms;
    ms.kernel_name = "v1_legacy";  // v1 backward-compat field
    REQUIRE(ms.kernels.empty());    // v2 vector field
    KernelEntry e;
    e.name = "v2_kernel";
    ms.kernels.push_back(e);
    REQUIRE(ms.kernels.size() == 1);
    REQUIRE(ms.kernels[0].name == "v2_kernel");
}
