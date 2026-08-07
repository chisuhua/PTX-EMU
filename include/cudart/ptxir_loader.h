#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>
#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/statement_context.h"

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

}  // namespace cudart
