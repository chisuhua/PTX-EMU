#ifndef MISC_HANDLER_H
#define MISC_HANDLER_H

#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/instruction_processor_utils.h"
#include "ptxsim/ptx_debug.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include <vector>

// MOV指令处理器
class MOV : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    virtual void process_operation(ThreadContext *context, 
                                  void* dst, void* src,
                                  std::vector<Qualifier>& qualifiers);
};

// SETP指令处理器
class SETP : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    virtual void process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2,
                                   std::vector<Qualifier> &qualifiers);
};

// ABS指令处理器
class ABS : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    virtual void process_operation(ThreadContext *context, void *dst, void *src,
                                   std::vector<Qualifier> &qualifiers);
};

// MIN指令处理器
class MIN : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    virtual void process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2,
                                   std::vector<Qualifier> &qualifiers);
};

// MAX指令处理器
class MAX : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    virtual void process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2,
                                   std::vector<Qualifier> &qualifiers);
};

// RCP指令处理器
class RCP : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    virtual void process_operation(ThreadContext *context, void *dst, void *src,
                                   std::vector<Qualifier> &qualifiers);
};

// NEG指令处理器
class NEG : public InstructionHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    virtual void process_operation(ThreadContext *context, void *dst, void *src,
                                   std::vector<Qualifier> &qualifiers);
};

#endif // MISC_HANDLER_H