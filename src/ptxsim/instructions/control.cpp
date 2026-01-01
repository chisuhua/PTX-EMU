#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

void BRA::process_operation(ThreadContext *context, void *op[1],
                            const std::vector<Qualifier> &qualifier) {
    int target_pc = *(int *)(op[0]);
    context->next_pc = target_pc - 1;
}

// void RET::process_operation(ThreadContext *context) { context->state = EXIT;
// }

// void BAR::process_operation(ThreadContext *context, BAR_TYPE bar_type) {
//     if (bar_type == BAR_TYPE::SYNC) {
//         context->state = BAR_SYNC;
//     }
// }