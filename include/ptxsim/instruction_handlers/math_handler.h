#ifndef MATH_HANDLER_H
#define MATH_HANDLER_H

#include "ptxsim/instruction_factory.h"
#include "ptxsim/thread_context.h"
#include "ptx_ir/ptx_types.h"
#include <vector>
#include <cmath>

// SQRT指令处理器
class SQRT : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// SIN指令处理器
class SIN : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

// COS指令处理器
class COS : public InstructionHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
};

#endif // MATH_HANDLER_H