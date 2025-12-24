#include "memory/memory_manager.h" // 确保包含 MemoryManager
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

void WMMA::process_operation(ThreadContext *context, void *op[4],
                             const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];
    void *src3 = op[3];
    // TODO
}