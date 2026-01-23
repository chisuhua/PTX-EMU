#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <iostream>

void MOV::process_operation(ThreadContext *context, void *op[2],
                            const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];

    context->mov(src, dst, qualifiers);
}

void CVTA::process_operation(ThreadContext *context, void *op[2],
                             const std::vector<Qualifier> &qualifiers) {
    void *to = op[0];
    void *from = op[1];

    // context->mov(from, to, qualifier);
    //  空指针检查
    if (!to || !from) {
        std::cerr << "Error: Null pointer in CVTA instruction" << std::endl;
        return;
    }

    // CVTA 是指针赋值：*to = *(void**)from
    // 即：将 from 指向的指针值，写入 to 指向的位置
    *(void **)to = *(void **)from;
}