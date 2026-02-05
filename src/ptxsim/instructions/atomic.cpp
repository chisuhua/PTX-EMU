#include "memory/hardware_memory_manager.h" // 添加HardwareMemoryManager头文件
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

void ATOM_Handler::processOperation(ThreadContext *context, void *op[3],
                             const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];
    // TODO
}