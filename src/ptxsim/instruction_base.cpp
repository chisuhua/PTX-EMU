#include "ptxsim/instruction_base.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/thread_context.h"

void InstructionHandler::execute_full(ThreadContext *context,
                                      StatementContext &stmt) {
    if (!prepare(context, stmt)) {
        return;
    }
    if (!execute(context, stmt)) {
        return;
    }
    commit(context, stmt);
    context->pc++;
}
