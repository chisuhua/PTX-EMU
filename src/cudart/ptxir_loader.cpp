#include "cudart/ptxir_loader.h"
#include "cudart/ptx_context_adapter.h"
#include "ptxir/ptxir_serialization.h"
#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/ptxir_reader.h"
#include <algorithm>
#include <cstring>
#include <openssl/sha.h>
#include <sstream>
#include <stdexcept>

namespace cudart {

ManifestSection read_manifest_from_ptxir_section(const uint8_t* data, size_t size) {
    if (!data || size == 0) {
        return ManifestSection();
    }
    try {
        std::string s(reinterpret_cast<const char*>(data), size);
        std::istringstream iss(s, std::ios::binary);
        PtxirReader reader(iss);
        reader.read();
        ManifestSection manifest = reader.get_manifest();
        // Backward-compat: if kernels vector is empty but kernel_name is set,
        // synthesize a single-entry vector (per ADR-0028 + ADR-0023 Extend-Only).
        if (manifest.kernels.empty() && !manifest.kernel_name.empty()) {
            KernelEntry entry;
            entry.name = manifest.kernel_name;
            manifest.kernels.push_back(entry);
        }
        return manifest;
    } catch (...) {
        return ManifestSection();
    }
}

// Phase 12.2 R3: testable dispatch helper that distinguishes "no footer"
// (caller may fallback to cuobjdump) from "footer present + malformed"
// (caller MUST NOT fallback — explicit error per 架构 §4.1 + ADR-0024 #6).
PtxirDispatchStatus try_ptxir_dispatch_from_memory(
    const uint8_t* exe_data, size_t exe_size,
    PtxContext* out_ctx) {

    // Step 1: detect PTXIR footer presence (cheap O(1) magic check).
    if (!PTXIRLoader::hasEmbeddedPTXIR(exe_data, exe_size)) {
        return PtxirDispatchStatus::kNoFooter;
    }

    // Step 2: extract PTXIR section. Footer is present but extract may fail
    // (e.g. size_le field points outside the buffer). Return explicit error.
    size_t section_size = 0;
    auto section = PTXIRLoader::extractPTXIR(exe_data, exe_size, &section_size);
    if (!section) {
        return PtxirDispatchStatus::kMalformedPtxir;
    }

    // Step 3: deserialize. Empty result = corrupted PTXIR section.
    auto stmts = PTXIRLoader::deserializeForCubin(section.get(), section_size);
    if (stmts.empty()) {
        return PtxirDispatchStatus::kMalformedPtxir;
    }

    // Step 4: validate manifest. Empty kernel_name is invalid per ADR-0024 §1.
    auto manifest = read_manifest_from_ptxir_section(section.get(), section_size);
    if (manifest.kernel_name.empty()) {
        return PtxirDispatchStatus::kMalformedManifest;
    }

    // Step 5: build PtxContext. Caller takes ownership via set_ptx_context.
    EmbeddedKernelManifest em;
    em.kernelName = manifest.kernel_name;
    em.ptxAddressSize = manifest.ptx_address_size;
    em.params = manifest.params;
    *out_ctx = PtxContextAdapter::fromEmbedded(std::move(stmts), em);
    return PtxirDispatchStatus::kSuccess;
}

namespace {

uint32_t read_le32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

std::vector<uint8_t> sha256(const uint8_t* data, size_t size) {
    std::vector<uint8_t> hash(32);
    if (data && size > 0) {
        SHA256(data, size, hash.data());
    }
    return hash;
}

ManifestSection read_manifest_from_ptxir(const uint8_t* data, size_t size) {
    try {
        std::string s(reinterpret_cast<const char*>(data), size);
        std::istringstream iss(s, std::ios::binary);
        PtxirReader reader(iss);
        reader.read();
        return reader.get_manifest();
    } catch (...) {
        return ManifestSection();
    }
}

}  // namespace

bool PTXIRLoader::hasEmbeddedPTXIR(const uint8_t* data, size_t size) {
    if (!data || size < 12) return false;
    if (std::memcmp(data + size - 8, PTXIR_EMBED_MAGIC, 8) != 0) return false;
    uint32_t section_size = read_le32(data + size - 12);
    if (size < 12 + section_size) return false;
    return true;
}

std::unique_ptr<uint8_t[]> PTXIRLoader::extractPTXIR(const uint8_t* data, size_t size, size_t* out_size) {
    if (!hasEmbeddedPTXIR(data, size)) {
        if (out_size) *out_size = 0;
        return nullptr;
    }
    uint32_t section_size = read_le32(data + size - 12);
    auto buf = std::make_unique<uint8_t[]>(section_size);
    std::memcpy(buf.get(), data + size - 12 - section_size, section_size);
    if (out_size) *out_size = section_size;
    return buf;
}

std::optional<std::vector<uint8_t>> PTXIRLoader::extractPureCubin(const uint8_t* data, size_t size) {
    if (!hasEmbeddedPTXIR(data, size)) {
        if (!data) return std::nullopt;
        return std::vector<uint8_t>(data, data + size);
    }
    uint32_t section_size = read_le32(data + size - 12);
    size_t prefix_size = size - 12 - section_size;
    std::vector<uint8_t> prefix(data, data + prefix_size);
    auto section = extractPTXIR(data, size, nullptr);
    if (section) {
        auto manifest = read_manifest_from_ptxir_section(section.get(), section_size);
        if (!manifest.cubin_hash.empty() &&
            std::all_of(manifest.cubin_hash.begin(), manifest.cubin_hash.end(),
                        [](uint8_t b) { return b != 0; })) {
            auto hash = sha256(prefix.data(), prefix.size());
            if (hash != manifest.cubin_hash) {
                return std::nullopt;
            }
        }
    }
    return prefix;
}

std::vector<ptxemu::ir::StatementContext> PTXIRLoader::deserializeForCubin(const uint8_t* ptxir_data, size_t ptxir_size) {
    std::vector<ptxemu::ir::StatementContext> result;
    if (!ptxir_data || ptxir_size == 0) return result;
    try {
        std::string s(reinterpret_cast<const char*>(ptxir_data), ptxir_size);
        result = deserialize_from_string(s);
    } catch (...) {
        return result;
    }
    return result;
}

}  // namespace cudart
