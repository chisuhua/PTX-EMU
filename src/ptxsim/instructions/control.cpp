#include "ptxsim/instruction_handlers.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include "ptxsim/warp_context.h"
#include <cmath>

void BRA::process_operation(ThreadContext *context, void *op[1],
                            const std::vector<Qualifier> &qualifier) {
    int target_pc = *(int *)(op[0]);
    context->next_pc = target_pc;
}

void AT::process_operation(ThreadContext *context, void *op[2],
                           const std::vector<Qualifier> &qualifier) {
    int8_t predicate = *(int8_t *)(op[0]);
    int target_pc = *(int *)(op[1]);
    if (predicate)
        context->next_pc = target_pc;
}

// void RET::process_operation(ThreadContext *context) { context->state = EXIT;
// }

void BAR::process_operation(ThreadContext *context, int barId,
                            const std::vector<Qualifier> &qualifier) {
    // 获取线程所属的warp上下文
    context->state = BAR_SYNC;
    context->bar_id = barId;
    context->next_pc = context->pc + 1;
    return; // 如果成功调用了同步方法，直接返回
}