#ifndef PTX_CONTEXT_H
#define PTX_CONTEXT_H

#include "kernel_context.h"
#include <vector>

class PtxContext {
public:
    int ptxMajorVersion;
    int ptxMinorVersion;
    int ptxTarget;
    int ptxAddressSize;

    std::vector<KernelContext> ptxKernels;
    std::vector<StatementContext> ptxStatements;

    PtxContext()
        : ptxMajorVersion(0), ptxMinorVersion(0), ptxTarget(0),
          ptxAddressSize(0) {}
};

#endif // KERNEL_CONTEXT_H