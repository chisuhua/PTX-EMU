#include "ptxsim/instruction_handlers/memory_handler.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include <cassert>
#include <iostream>
#include <vector>

void LdHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::LD *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->ldOp[0], ss->ldQualifier);
    void *from = context->get_operand_addr(ss->ldOp[1], ss->ldQualifier);

    // 添加空指针检查，防止段错误
    if (!to || !from) {
        std::cerr << "Error: Null pointer dereference in LD instruction"
                  << std::endl;
        return;
    }

    // 获取数据大小
    size_t data_size = TypeUtils::get_bytes(ss->ldQualifier);

    // 获取寄存器信息用于跟踪
    OperandContext::REG *reg_operand = nullptr;
    if (ss->ldOp[0].operandType == OperandType::O_REG) {
        reg_operand = static_cast<OperandContext::REG *>(ss->ldOp[0].operand);
    }

    // 执行LD操作，包括内存读取跟踪和实际数据移动
    std::string addr_expr = ss->ldOp[1].toString();
    context->memory_access(false, addr_expr, from, data_size, nullptr,
                           ss->ldQualifier, to, reg_operand);

    // 处理向量操作
    if (context->QvecHasQ(ss->ldQualifier, Qualifier::Q_V2)) {
        uint64_t step = context->getBytes(ss->ldQualifier);
        auto vecAddr = context->vec.front()->vec;
        context->vec.pop();
        assert(vecAddr.size() == 2);
        for (int i = 0; i < 2; i++) {
            to = vecAddr[i];
            void *src_addr = (void *)((uint64_t)from + i * step);
            // 为每个向量元素添加内存读取跟踪和实际数据移动
            context->memory_access(
                false, addr_expr + "[" + std::to_string(i) + "]", src_addr,
                step, nullptr, ss->ldQualifier, to, reg_operand);
        }
    } else if (context->QvecHasQ(ss->ldQualifier, Qualifier::Q_V4)) {
        uint64_t step = context->getBytes(ss->ldQualifier);
        auto vecAddr = context->vec.front()->vec;
        context->vec.pop();
        assert(vecAddr.size() == 4);
        for (int i = 0; i < 4; i++) {
            to = vecAddr[i];
            void *src_addr = (void *)((uint64_t)from + i * step);
            // 为每个向量元素添加内存读取跟踪和实际数据移动
            context->memory_access(
                false, addr_expr + "[" + std::to_string(i) + "]", src_addr,
                step, nullptr, ss->ldQualifier, to, reg_operand);
        }
    }
}

void StHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::ST *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->stOp[0], ss->stQualifier);
    void *from = context->get_operand_addr(ss->stOp[1], ss->stQualifier);

    // 添加空指针检查，防止段错误
    if (!to || !from) {
        std::cerr << "Error: Null pointer dereference in ST instruction"
                  << std::endl;
        return;
    }

    // 获取数据大小
    size_t data_size = TypeUtils::get_bytes(ss->stQualifier);

    // 获取寄存器信息用于跟踪
    OperandContext::REG *reg_operand = nullptr;
    if (ss->stOp[1].operandType == OperandType::O_REG) {
        reg_operand = static_cast<OperandContext::REG *>(ss->stOp[1].operand);
    }

    // 执行ST操作，包括内存写入跟踪和实际数据移动
    std::string addr_expr = ss->stOp[0].toString();
    context->memory_access(true, addr_expr, to, data_size, from,
                           ss->stQualifier, nullptr, reg_operand);

    // 处理向量操作
    if (context->QvecHasQ(ss->stQualifier, Qualifier::Q_V4)) {
        uint64_t step = context->getBytes(ss->stQualifier);
        auto vecAddr = context->vec.front()->vec;
        context->vec.pop();
        assert(vecAddr.size() == 4);
        for (int i = 0; i < 4; i++) {
            from = vecAddr[i];
            void *dst_addr = (void *)((uint64_t)to + i * step);
            // 为每个向量元素添加内存写入跟踪和实际数据移动
            context->memory_access(
                true, addr_expr + "[" + std::to_string(i) + "]", dst_addr, step,
                from, ss->stQualifier, nullptr, reg_operand);
        }
    } else if (context->QvecHasQ(ss->stQualifier, Qualifier::Q_V2)) {
        uint64_t step = context->getBytes(ss->stQualifier);
        auto vecAddr = context->vec.front()->vec;
        context->vec.pop();
        assert(vecAddr.size() == 2);
        for (int i = 0; i < 2; i++) {
            from = vecAddr[i];
            void *dst_addr = (void *)((uint64_t)to + i * step);
            // 为每个向量元素添加内存写入跟踪和实际数据移动
            context->memory_access(
                true, addr_expr + "[" + std::to_string(i) + "]", dst_addr, step,
                from, ss->stQualifier, nullptr, reg_operand);
        }
    }
}