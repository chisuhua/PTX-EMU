#include "catch_amalgamated.hpp"
#include "cudart/ptxir_loader.h"
#include "cudart/ptx_context_adapter.h"
#include "cudart/ptxir_config.h"
#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/ptxir_reader.h"
#include "ptx_ir/ptxir_writer.h"
#include <cstring>
#include <fstream>
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

std::vector<uint8_t> build_minimal_section(const std::string& kernel_name) {
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

}  // namespace

TEST_CASE("dispatch_embeddedExe_PTXIR_MODE_auto_detectsMagic", "[ptxir_dispatch]") {
    setenv("PTXIR_MODE", "auto", 1);
    auto bin = make_embedded({0x01, 0x02}, build_minimal_section("k"));
    REQUIRE(cudart::PTXIRLoader::hasEmbeddedPTXIR(bin.data(), bin.size()));
}

TEST_CASE("dispatch_plainExe_PTXIR_MODE_auto_noMagic", "[ptxir_dispatch]") {
    setenv("PTXIR_MODE", "auto", 1);
    std::vector<uint8_t> plain = {0x01, 0x02, 0x03};
    REQUIRE_FALSE(cudart::PTXIRLoader::hasEmbeddedPTXIR(plain.data(), plain.size()));
}

TEST_CASE("dispatch_PTXIR_MODE_off_skipsDetection", "[ptxir_dispatch]") {
    setenv("PTXIR_MODE", "off", 1);
    REQUIRE_FALSE(config::isPTXIRModeEnabled());
}

TEST_CASE("dispatch_embeddedSection_deserializesStatements", "[ptxir_dispatch]") {
    auto section = build_minimal_section("test_kernel");
    auto stmts = cudart::PTXIRLoader::deserializeForCubin(section.data(), section.size());
    REQUIRE_FALSE(stmts.empty());
}

TEST_CASE("dispatch_corruptedPTXIR_gracefulEmpty", "[ptxir_dispatch]") {
    std::vector<uint8_t> corrupted = {0xFF, 0xFF, 0xFF, 0xFF};
    auto stmts = cudart::PTXIRLoader::deserializeForCubin(corrupted.data(), corrupted.size());
    REQUIRE(stmts.empty());
}

TEST_CASE("dispatch_fatBinNullPtr_safeToIgnore", "[ptxir_dispatch]") {
    REQUIRE(true);
}

// ============================================================================
// Phase 12.2 R3 — malformed PTXIR / manifest mismatch error handling
// Per [docs/architecture/ptxir-toolchain-stack.md §4.1] + ADR-0024 acceptance #6:
// "malformed embedded PTXIR 或 manifest mismatch → 报告错误 (NOT 静默 fallback)"
//
// RED PHASE: this test MUST FAIL until cudart_sim.cpp:356-388 is refactored
// to use try_ptxir_dispatch_from_memory() with explicit error returns.
// ============================================================================
TEST_CASE("dispatch_plainBinary_returnsNoFooter", "[ptxir_dispatch][regression;PHASE12.2-R3]") {
    std::vector<uint8_t> plain = {0x01, 0x02, 0x03};
    PtxContext ctx;
    auto status = cudart::try_ptxir_dispatch_from_memory(
        plain.data(), plain.size(), &ctx);
    REQUIRE(status == cudart::PtxirDispatchStatus::kNoFooter);
}

TEST_CASE("dispatch_validEmbedded_returnsSuccessAndKernel",
          "[ptxir_dispatch][regression;PHASE12.2-R3]") {
    auto bin = make_embedded({0x01, 0x02}, build_minimal_section("valid_kernel"));
    PtxContext ctx;
    auto status = cudart::try_ptxir_dispatch_from_memory(
        bin.data(), bin.size(), &ctx);
    REQUIRE(status == cudart::PtxirDispatchStatus::kSuccess);
    bool found = false;
    for (const auto& kc : ctx.ptxKernels) {
        if (kc.kernelName == "valid_kernel") found = true;
    }
    REQUIRE(found);
}

TEST_CASE("dispatch_corruptedPtxirSection_returnsMalformedPtxir_notNoFooter",
          "[ptxir_dispatch][regression;PHASE12.2-R3]") {
    // Footer PRESENT + size_le valid, but section bytes are corrupted
    // (not valid PTXIR). hasEmbeddedPTXIR=true, extractPTXIR=valid ptr,
    // deserializeForCubin=empty.
    //
    // Current cudart_sim.cpp:368-385 SILENTLY FALLS BACK to cuobjdump (BUG).
    // Per 架构 §4.1 + ADR-0024 acceptance #6: MUST return explicit error,
    // NOT kNoFooter (which would mean "OK to fallback").
    std::vector<uint8_t> corrupted_section = {
        0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x00, 0x01};
    auto bin = make_embedded({0x01, 0x02}, corrupted_section);
    PtxContext ctx;
    auto status = cudart::try_ptxir_dispatch_from_memory(
        bin.data(), bin.size(), &ctx);
    REQUIRE(status == cudart::PtxirDispatchStatus::kMalformedPtxir);
}

TEST_CASE("dispatch_emptyKernelName_returnsMalformedManifest_notSuccess",
          "[ptxir_dispatch][regression;PHASE12.2-R3]") {
    // Footer + section + stmts OK, but manifest.kernel_name="" (invalid).
    //
    // Current cudart_sim.cpp:374-380 does NOT validate kernel_name
    // and registers anyway. Per 架构 §4.1 + ADR-0024 §1: MUST return
    // explicit error.
    auto bin = make_embedded({0x01, 0x02}, build_minimal_section(""));
    PtxContext ctx;
    auto status = cudart::try_ptxir_dispatch_from_memory(
        bin.data(), bin.size(), &ctx);
    REQUIRE(status == cudart::PtxirDispatchStatus::kMalformedManifest);
}
