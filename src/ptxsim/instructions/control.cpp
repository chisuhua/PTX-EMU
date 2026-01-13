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

void BAR::process_operation(ThreadContext *context, int barId,
                            const std::vector<Qualifier> &qualifier) {
    // 注意：这里的实现需要能够访问SMContext来执行barrier同步
    // 由于ThreadContext不能直接访问SMContext，我们需要通过其他方式实现
    // 目前将线程状态设置为BAR_SYNC，然后在SMContext::exe_once中处理同步逻辑
    
    // 为了更好地实现barrier功能，我们先设置线程状态为BAR_SYNC
    context->state = BAR_SYNC;
}