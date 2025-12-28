#include "ptxsim/instruction_base.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/thread_context.h"

void INSTR_BASE::execute(ThreadContext *context, StatementContext &stmt) {
    if (stmt.state == InstructionState::READY) {
        if (!prepare(context, stmt)) {
            return;
        }
        stmt.state = InstructionState::PREPARE;
    }
    if (stmt.state == InstructionState::PREPARE) {
        if (!operate(context, stmt)) {
            return;
        }
        stmt.state = InstructionState::EXECUTE;
    }
    if (stmt.state == InstructionState::EXECUTE) {
        if (!commit(context, stmt)) {
            return;
        }
        stmt.state = InstructionState::COMMIT;
        context->pc++;
    }
}

bool INSTR_BASE::prepare(ThreadContext *context, StatementContext &stmt) {
    return true;
}
bool INSTR_BASE::commit(ThreadContext *context, StatementContext &stmt) {
    return true;
}

bool GENERIC_INSTR::operate(ThreadContext *context, StatementContext &stmt) {
    process_operation(context, &stmt.oc[0], *stmt.qualifier);
    return true;
}