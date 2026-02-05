#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <iostream>

void MOV_Handler::processOperation(ThreadContext *context, void **operands,
                                   const std::vector<Qualifier> &qualifiers) {
    void *dst = operands[0];
    void *src = operands[1];

    context->mov(src, dst, qualifiers);
}

void CVTA_Handler::processOperation(ThreadContext *context, void **operands,
                                    const std::vector<Qualifier> &qualifiers) {
    void *to = operands[0];
    void *from = operands[1];

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
