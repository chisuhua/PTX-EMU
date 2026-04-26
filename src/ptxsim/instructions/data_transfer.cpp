#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <iostream>

void MovHandler::processOperation(ThreadContext *context, void **operands,
                                   const std::vector<Qualifier> &qualifiers,
                                   const std::vector<char> *operand_is_immediate) {
    void *dst = operands[0];
    void *src = operands[1];

    (void)operand_is_immediate;
    context->mov(src, dst, qualifiers);
}

void CvtaHandler::processOperation(ThreadContext *context, void **operands,
                                    const std::vector<Qualifier> &qualifiers,
                                    const std::vector<char> *operand_is_immediate) {
    void *to = operands[0];
    void *from = operands[1];

    // 空指针检查
    if (!to || !from) {
        std::cerr << "Error: Null pointer in CVTA instruction" << std::endl;
        return;
    }
    
    // CVTA 是指针赋值：*(void**)to = *(void**)from
    // 即：将 from 指向的指针值，写入 to 指向的位置
    *(void **)to = *(void **)from;
}
