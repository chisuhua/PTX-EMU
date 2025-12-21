#ifndef BIT_MANIPULATE_H
#define BIT_MANIPULATE_H

#include "ptxsim/instruction_factory.h"
#include "ptxsim/ptx_debug.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include <vector>

class POPC : public InstructionHandler {
public:
    virtual void execute(ThreadContext *context, StatementContext &stmt);

protected:
    virtual void process_operation(ThreadContext *context, void *dst,
                                   void *src1,
                                   std::vector<Qualifier> &qualifiers);
};

class CLZ : public InstructionHandler {
public:
    virtual void execute(ThreadContext *context, StatementContext &stmt);

protected:
    virtual void process_operation(ThreadContext *context, void *dst,
                                   void *src1,
                                   std::vector<Qualifier> &qualifiers);
};

#endif