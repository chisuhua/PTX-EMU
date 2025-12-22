#include "ptxsim/instruction_handlers_decl.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

void BRA::process_operation(ThreadContext *context, void *op[1],
                            std::vector<Qualifier> &qualifier) {
    int target_pc = *(int *)(op[0]);
    context->pc = target_pc - 1;
}

// void RET::process_operation(ThreadContext *context) { context->state = EXIT;
// }

// void BAR::process_operation(ThreadContext *context, BAR_TYPE bar_type) {
//     if (bar_type == BAR_TYPE::SYNC) {
//         context->state = BAR_SYNC;
//     }
// }
