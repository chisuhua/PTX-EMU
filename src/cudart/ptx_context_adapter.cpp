#include "cudart/ptx_context_adapter.h"
#include "ptx_ir/kernel_context.h"
#include "ptx_ir/ptx_types.h"

namespace cudart {

PtxContext PtxContextAdapter::fromEmbedded(std::vector<ptxemu::ir::StatementContext> stmts,
                                            const EmbeddedKernelManifest& manifest) {
    KernelContext kc;
    kc.kernelName = manifest.kernelName;
    kc.ifEntryKernel = true;
    kc.kernelStatements = std::move(stmts);

    for (const auto& p : manifest.params) {
        ParamContext pc;
        pc.paramName = p.name;
        pc.byteSize = p.size;
        switch (p.kind) {
            case ParamKind::U8:
                pc.paramTypes.push_back(ptxemu::ir::Qualifier::Q_U8);
                break;
            case ParamKind::U16:
                pc.paramTypes.push_back(ptxemu::ir::Qualifier::Q_U16);
                break;
            case ParamKind::U32:
                pc.paramTypes.push_back(ptxemu::ir::Qualifier::Q_U32);
                break;
            case ParamKind::U64:
                pc.paramTypes.push_back(ptxemu::ir::Qualifier::Q_U64);
                break;
            case ParamKind::F32:
                pc.paramTypes.push_back(ptxemu::ir::Qualifier::Q_F32);
                break;
            case ParamKind::F64:
                pc.paramTypes.push_back(ptxemu::ir::Qualifier::Q_F64);
                break;
        }
        if (p.kind == ParamKind::U64 && p.size == 8) {
            pc.isPtr = true;
            pc.paramTypes.push_back(ptxemu::ir::Qualifier::Q_PTR);
        }
        kc.kernelParams.push_back(pc);
    }

    PtxContext ctx;
    ctx.ptxAddressSize = manifest.ptxAddressSize;
    ctx.ptxKernels.push_back(std::move(kc));
    return ctx;
}

}  // namespace cudart
