#ifndef KERNEL_CONTEXT_H
#define KERNEL_CONTEXT_H

#include "statement_context.h"
#include <string>
#include <vector>

class ParamContext {
public:
    Qualifier paramType;
    std::string paramName;
    int paramAlign;
    int paramNum;

    ParamContext() : paramType(Qualifier::Q_B32), paramAlign(0), paramNum(0) {}
};

class KernelContext {
public:
    bool ifVisibleKernel;
    bool ifEntryKernel;
    std::string kernelName;

    class Maxntid {
    public:
        int x, y, z;
        Maxntid() : x(0), y(0), z(0) {}
    };

    Maxntid maxntid;
    int minnctapersm;

    std::vector<ParamContext> kernelParams;
    std::vector<StatementContext> kernelStatements;

    KernelContext()
        : ifVisibleKernel(false), ifEntryKernel(false), minnctapersm(0) {}
};

#endif // KERNEL_CONTEXT_H