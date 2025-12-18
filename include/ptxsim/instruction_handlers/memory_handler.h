#ifndef MEMORY_HANDLER_H
#define MEMORY_HANDLER_H

#include "ptxsim/instruction_factory.h"
#include "ptxsim/thread_context.h"
#include "ptx_ir/ptx_types.h"
#include <vector>

// LD指令处理器
class LD : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// ST指令处理器
class ST : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

#endif // MEMORY_HANDLER_H