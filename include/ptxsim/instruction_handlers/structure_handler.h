#ifndef STRUCTURE_HANDLER_H
#define STRUCTURE_HANDLER_H

#include "ptxsim/instruction_factory.h"
#include "ptxsim/thread_context.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/operand_context.h"
#include <vector>

// REG指令处理器
class RegHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// SHARED指令处理器
class SharedHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// LOCAL指令处理器
class LocalHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// DOLLOR指令处理器
class DollorHandler : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

#endif // STRUCTURE_HANDLER_H