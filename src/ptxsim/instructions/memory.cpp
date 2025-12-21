#include "ptxsim/instruction_handlers_decl.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

void MOV::process_operation(ThreadContext *context, void *op[2],
                            std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];

    context->mov(src, dst, qualifiers);
}

void LD::process_operation(ThreadContext *context, void *op[2],
                           std::vector<Qualifier> &qualifier) {
    void *to = op[0];
    void *from = op[1];
    // 添加空指针检查，防止段错误
    if (!to || !from) {
        std::cerr << "Error: Null pointer dereference in LD instruction"
                  << std::endl;
        return;
    }

    // 获取数据大小
    size_t data_size = TypeUtils::get_bytes(qualifier);

    // 获取寄存器信息用于跟踪
    // OperandContext::REG *reg_operand = nullptr;
    // if (ss->ldOp[0].operandType == OperandType::O_REG) {
    //     reg_operand = static_cast<OperandContext::REG
    //     *>(ss->ldOp[0].operand);
    // }

    // 执行LD操作，包括内存读取跟踪和实际数据移动
    std::string addr_expr = ss->ldOp[1].toString();
    context->memory_access(false, addr_expr, from, data_size, nullptr,
                           ss->ldQualifier, to, reg_operand);

    // 处理向量操作
    if (context->QvecHasQ(qualifier, Qualifier::Q_V2)) {
        uint64_t step = context->getBytes(qualifier);
        auto vecAddr = context->vec.front()->vec;
        context->vec.pop();
        assert(vecAddr.size() == 2);
        for (int i = 0; i < 2; i++) {
            to = vecAddr[i];
            void *src_addr = (void *)((uint64_t)from + i * step);
            // 为每个向量元素添加内存读取跟踪和实际数据移动
            context->memory_access(
                false, addr_expr + "[" + std::to_string(i) + "]", src_addr,
                step, nullptr, qualifier, to, reg_operand);
        }
    } else if (context->QvecHasQ(qualifier, Qualifier::Q_V4)) {
        uint64_t step = context->getBytes(qualifier);
        auto vecAddr = context->vec.front()->vec;
        context->vec.pop();
        assert(vecAddr.size() == 4);
        for (int i = 0; i < 4; i++) {
            to = vecAddr[i];
            void *src_addr = (void *)((uint64_t)from + i * step);
            // 为每个向量元素添加内存读取跟踪和实际数据移动
            context->memory_access(
                false, addr_expr + "[" + std::to_string(i) + "]", src_addr,
                step, nullptr, qualifier, to, reg_operand);
        }
    }
}

void ST::process_operation(ThreadContext *context, void *op[2],
                           std::vector<Qualifier> &qualifiers) {
    void *to = op[0];
    void *from = op[1];
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
