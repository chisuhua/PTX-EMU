#pragma once
#include "ptx_ir/ptx_context.h"
#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/statement_context.h"
#include <string>
#include <vector>

namespace cudart {

struct EmbeddedKernelManifest {
    std::string kernelName;
    std::vector<ManifestParam> params;
    int ptxAddressSize = 64;
};

class PtxContextAdapter {
public:
    static PtxContext fromEmbedded(std::vector<StatementContext> stmts,
                                   const EmbeddedKernelManifest& manifest);
};

}  // namespace cudart
