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
class CVT : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    virtual void process_operation(ThreadContext *context, void *dst, void *src,
                                   std::vector<Qualifier> &qualifiers);
};

// CVTA指令处理器
class CVTA : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// SELP指令处理器
class SELP : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    virtual void process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2, void *pred,
                                   std::vector<Qualifier> &qualifiers);
};

// NOT指令处理器
class NOT : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// REM指令处理器
class REM : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// RSQRT指令处理器
class RSQRT : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// LG2指令处理器
class LG2 : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// EX2指令处理器
class EX2 : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

// WMMA指令处理器
class WMMA : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;
};

#endif // REMAINING_HANDLER_H