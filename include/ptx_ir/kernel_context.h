#ifndef KERNEL_CONTEXT_H
#define KERNEL_CONTEXT_H

#include "statement_context.h"
#include <string>
#include <vector>

class ParamContext {
public:
    std::vector<Qualifier> paramTypes;  // 修改为支持多个类型
    std::string paramName;
    int paramAlign;
    int paramNum;
    bool isPtr;  // 新增字段，用于存储PTR修饰符

    ParamContext() : paramAlign(0), paramNum(0), isPtr(false) {}
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