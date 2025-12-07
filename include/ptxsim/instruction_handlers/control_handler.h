#ifndef CONTROL_HANDLER_H
#define CONTROL_HANDLER_H

#include "ptxsim/instruction_factory.h"
#include "ptxsim/thread_context.h"
#include "ptx_ir/ptx_types.h"
#include <vector>

// BRA指令处理器
class BraHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// RET指令处理器
class RetHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// BAR指令处理器
class BarHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

#endif // CONTROL_HANDLER_H