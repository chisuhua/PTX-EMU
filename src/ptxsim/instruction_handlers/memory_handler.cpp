#include "ptxsim/instruction_handlers/memory_handler.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include "ptx_ir/ptx_types.h"
#include <cassert>
#include <vector>
#include <iostream>

void LdHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::LD*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->ldOp[0], ss->ldQualifier);
    void *from = context->get_operand_addr(ss->ldOp[1], ss->ldQualifier);
    
    // 添加空指针检查，防止段错误
    if (!to || !from) {
        std::cerr << "Error: Null pointer dereference in LD instruction" << std::endl;
        return;
    }
    
    // 执行LD操作
    // 注意：这里需要通过公共接口访问ThreadContext的私有成员
    if (context->QvecHasQ(ss->ldQualifier, Qualifier::Q_V2)) {
        uint64_t step = context->getBytes(ss->ldQualifier);
        auto vecAddr = context->vec.front()->vec;
        context->vec.pop();
        assert(vecAddr.size() == 2);
        for (int i = 0; i < 2; i++) {
            to = vecAddr[i];
            context->mov((void *)((uint64_t)from + i * step), to, ss->ldQualifier);
        }
    } else if (context->QvecHasQ(ss->ldQualifier, Qualifier::Q_V4)) {
        uint64_t step = context->getBytes(ss->ldQualifier);
        auto vecAddr = context->vec.front()->vec;
        context->vec.pop();
        assert(vecAddr.size() == 4);
        for (int i = 0; i < 4; i++) {
            to = vecAddr[i];
            context->mov((void *)((uint64_t)from + i * step), to, ss->ldQualifier);
        }
    } else {
        context->mov(from, to, ss->ldQualifier);
    }
}

void StHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::ST*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->stOp[0], ss->stQualifier);
    void *from = context->get_operand_addr(ss->stOp[1], ss->stQualifier);
    
    // 添加空指针检查，防止段错误
    if (!to || !from) {
        std::cerr << "Error: Null pointer dereference in ST instruction" << std::endl;
        return;
    }
    
    // 执行ST操作
    // 注意：这里需要通过公共接口访问ThreadContext的私有成员
    if (context->QvecHasQ(ss->stQualifier, Qualifier::Q_V4)) {
        uint64_t step = context->getBytes(ss->stQualifier);
        auto vecAddr = context->vec.front()->vec;
        context->vec.pop();
        assert(vecAddr.size() == 4);
        for (int i = 0; i < 4; i++) {
            from = vecAddr[i];
            context->mov(from, (void *)((uint64_t)to + i * step), ss->stQualifier);
        }
    } else if (context->QvecHasQ(ss->stQualifier, Qualifier::Q_V2)) {
        uint64_t step = context->getBytes(ss->stQualifier);
        auto vecAddr = context->vec.front()->vec;
        context->vec.pop();
        assert(vecAddr.size() == 2);
        for (int i = 0; i < 2; i++) {
            from = vecAddr[i];
            context->mov(from, (void *)((uint64_t)to + i * step), ss->stQualifier);
        }
    } else {
        context->mov(from, to, ss->stQualifier);
    }
}