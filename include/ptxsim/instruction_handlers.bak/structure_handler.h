#ifndef STRUCTURE_HANDLER_H
#define STRUCTURE_HANDLER_H

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/thread_context.h"
#include <vector>

// REG指令处理器
class REG : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// SHARED指令处理器
class SHARED : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// LOCAL指令处理器
class LOCAL : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// DOLLOR指令处理器
class DOLLOR : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

class CONST : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

#endif // STRUCTURE_HANDLER_H