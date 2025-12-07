#ifndef MISC_HANDLER_H
#define MISC_HANDLER_H

#include "ptxsim/instruction_factory.h"
#include "ptxsim/thread_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include <vector>

// MOV指令处理器
class MovHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// SETP指令处理器
class SetpHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// CVT指令处理器
class CvtHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// ABS指令处理器
class AbsHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// MIN指令处理器
class MinHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// MAX指令处理器
class MaxHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// RCP指令处理器
class RcpHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// NEG指令处理器
class NegHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// MAD指令处理器
class MadHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// FMA指令处理器
class FmaHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

#endif // MISC_HANDLER_H