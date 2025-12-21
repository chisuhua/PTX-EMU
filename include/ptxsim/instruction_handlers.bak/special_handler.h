#ifndef SPECIAL_HANDLER_H
#define SPECIAL_HANDLER_H

#include "ptxsim/instruction_factory.h"
#include "ptxsim/thread_context.h"
#include "ptx_ir/ptx_types.h"
#include <vector>

// PRAGMA指令处理器
class PRAGMA : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// AT指令处理器
class AT : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// ATOM指令处理器
class ATOM : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

#endif // SPECIAL_HANDLER_H