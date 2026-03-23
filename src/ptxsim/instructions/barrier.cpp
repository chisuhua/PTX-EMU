#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/execution_types.h"
#include "utils/logger.h"

void BarHandler::executeBarrier(ThreadContext *context, const BarrierInstr &instr) {
    int bar_id = instr.barId.has_value() ? instr.barId.value() : 0;
    context->bar_id = bar_id;
    context->state = BAR_SYNC;
    context->next_pc = context->pc + 1;
    
    PTX_DEBUG_EMU("Thread (%d,%d,%d) waiting at bar.sync %d, state=BAR_SYNC, next_pc=%d",
                  context->ThreadIdx.x, context->ThreadIdx.y, context->ThreadIdx.z,
                  bar_id, context->next_pc);
}
