#ifndef REMAINING_HANDLER_H
#define REMAINING_HANDLER_H

#include "../../ptx_ir/operand_context.h"
#include "../../ptx_ir/ptx_types.h"
#include "../../ptx_ir/statement_context.h"
#include "../instruction_factory.h"
#include "../instruction_processor_utils.h"
#include "../thread_context.h"
#include <vector>

// CVT指令处理器
class CvtHandler : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    virtual void process_operation(ThreadContext *context, void *dst, void *src,
                                   std::vector<Qualifier> &qualifiers);
};

// CVTA指令处理器
class CvtaHandler : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// SELP指令处理器
class SelpHandler : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// NOT指令处理器
class NotHandler : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// REM指令处理器
class RemHandler : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// RSQRT指令处理器
class RsqrtHandler : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// LG2指令处理器
class Lg2Handler : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// EX2指令处理器
class Ex2Handler : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// WMMA指令处理器
class WmmaHandler : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

#endif // REMAINING_HANDLER_H