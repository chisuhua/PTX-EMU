#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>
#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/ptx_context.h"

namespace cudart {

inline constexpr uint8_t PTXIR_EMBED_MAGIC[8] = {'P','T','X','E','M','B','\x01','\x00'};

ManifestSection read_manifest_from_ptxir_section(const uint8_t* data, size_t size);

class PTXIRLoader {
public:
    static bool hasEmbeddedPTXIR(const uint8_t* data, size_t size);
    static std::unique_ptr<uint8_t[]> extractPTXIR(const uint8_t* data, size_t size, size_t* out_size);
    static std::optional<std::vector<uint8_t>> extractPureCubin(const uint8_t* data, size_t size);
    static std::vector<StatementContext> deserializeForCubin(const uint8_t* ptxir_data, size_t ptxir_size);
};

// ============================================================================
// PTXIR Dispatch Status (Phase 12.2 R3)
// ============================================================================
// Per [docs/architecture/ptxir-toolchain-stack.md §4.1] + ADR-0024 acceptance #6:
// "malformed embedded PTXIR or manifest mismatch → 报告错误 (NOT 静默 fallback)"
//
// This distinguishes the four outcomes of trying to dispatch an executable
// buffer via the embedded-PTXIR path. The caller (__cudaRegisterFatBinary)
// must use this to decide between:
//   - kSuccess: register the PtxContext and return success
//   - kNoFooter: no PTXIR footer present, OK to fallback to cuobjdump
//   - kMalformedPtxir / kMalformedManifest: explicit error, MUST NOT fallback
// ============================================================================
enum class PtxirDispatchStatus {
    kNoFooter,           // No PTXIR footer; caller SHOULD fallback to cuobjdump
    kSuccess,            // PtxContext populated from embedded PTXIR
    kMalformedPtxir,     // Footer present but extract/deserialize failed
    kMalformedManifest,  // Footer + section OK but manifest invalid (empty kernel_name)
};

// Testable pure dispatch helper (per R3 refactor).
// Reads exe_data[0..exe_size), checks for embedded PTXIR footer, and either:
//   - populates *out_ctx + returns kSuccess, OR
//   - returns kNoFooter (no footer present), OR
//   - returns kMalformedPtxir / kMalformedManifest (footer present + malformed).
//
// Caller MUST NOT fallback to cuobjdump on kMalformed* — this is the R3 fix
// for the silent-fallback bug in cudart_sim.cpp:356-388.
PtxirDispatchStatus try_ptxir_dispatch_from_memory(
    const uint8_t* exe_data, size_t exe_size,
    /* out */ PtxContext* out_ctx);

}  // namespace cudart
