// InstructionHandlers_decl.h
#ifndef INSTRUCTION_HANDLE_H
#define INSTRUCTION_HANDLE_H
#include "ptx_ir/statement_context.h"
#include <memory>
#include <vector>

class ThreadContext;

class InstructionHandler {
public:
    virtual ~InstructionHandler() = default;
    virtual void execute(ThreadContext *context, StatementContext &stmt) = 0;
    // virtual std::unique_ptr<InstructionHandler> clone() const {
    //     return std::make_unique<InstructionHandler>(*this);
    // }
};

#endif