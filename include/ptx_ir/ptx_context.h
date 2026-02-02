#ifndef PTX_CONTEXT_H
#define PTX_CONTEXT_H

#include "kernel_context.h"
#include <vector>

// ============ 新增：ExternFuncDecl ============
struct ExternFuncDecl {
    std::string name;
    std::vector<ParamContext> params; // 可选：记录参数信息
};

class PtxContext {
public:
    int ptxMajorVersion;
    int ptxMinorVersion;
    int ptxTarget;
    int ptxAddressSize;

    std::vector<KernelContext> ptxKernels;
    std::vector<StatementContext> ptxStatements;
    std::vector<ExternFuncDecl> externFuncs;

    PtxContext()
        : ptxMajorVersion(0), ptxMinorVersion(0), ptxTarget(0),
          ptxAddressSize(0) {}
};

#endif // KERNEL_CONTEXT_H