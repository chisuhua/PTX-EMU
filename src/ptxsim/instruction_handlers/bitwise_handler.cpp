#include "ptxsim/instruction_handlers/bitwise_handler.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include <cassert>

void BitwiseHandler::execute(ThreadContext* context, StatementContext& stmt) {
    // 这是一个抽象基类，不应该被直接调用
    assert(0 && "BitwiseHandler::execute should not be called directly");
}

void AndHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::AND*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->andOp[0], ss->andQualifier);
    void *op1 = context->get_operand_addr(ss->andOp[1], ss->andQualifier);
    void *op2 = context->get_operand_addr(ss->andOp[2], ss->andQualifier);
    
    // 执行AND操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->andQualifier, 
                        (OperandContext::REG*)ss->andOp[0].operand);
}

void OrHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::OR*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->orOp[0], ss->orQualifier);
    void *op1 = context->get_operand_addr(ss->orOp[1], ss->orQualifier);
    void *op2 = context->get_operand_addr(ss->orOp[2], ss->orQualifier);
    
    // 执行OR操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->orQualifier, 
                        (OperandContext::REG*)ss->orOp[0].operand);
}

void XorHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::XOR*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->xorOp[0], ss->xorQualifier);
    void *op1 = context->get_operand_addr(ss->xorOp[1], ss->xorQualifier);
    void *op2 = context->get_operand_addr(ss->xorOp[2], ss->xorQualifier);
    
    // 执行XOR操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->xorQualifier, 
                        (OperandContext::REG*)ss->xorOp[0].operand);
}

void ShlHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::SHL*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->shlOp[0], ss->shlQualifier);
    void *op1 = context->get_operand_addr(ss->shlOp[1], ss->shlQualifier);
    
    // 第三个操作数使用与目标相同的类型限定符
    void *op2 = context->get_operand_addr(ss->shlOp[2], ss->shlQualifier);
    
    // 执行SHL操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->shlQualifier, 
                        (OperandContext::REG*)ss->shlOp[0].operand);
}

void ShrHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::SHR*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->shrOp[0], ss->shrQualifier);
    void *op1 = context->get_operand_addr(ss->shrOp[1], ss->shrQualifier);
    
    // 第三个操作数使用与目标相同的类型限定符
    void *op2 = context->get_operand_addr(ss->shrOp[2], ss->shrQualifier);
    
    // 执行SHR操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->shrQualifier, 
                        (OperandContext::REG*)ss->shrOp[0].operand);
}

void AndHandler::process_operation(ThreadContext* context, 
                                 void* dst, void* src1, void* src2,
                                 std::vector<Qualifier>& qualifiers) {
    // 实现AND指令的具体逻辑
    int bytes = TypeUtils::get_bytes(qualifiers);
    
    switch (bytes) {
    case 1:
        *(uint8_t*)dst = (*(uint8_t*)src1) & (*(uint8_t*)src2);
        break;
    case 2:
        *(uint16_t*)dst = (*(uint16_t*)src1) & (*(uint16_t*)src2);
        break;
    case 4:
        *(uint32_t*)dst = (*(uint32_t*)src1) & (*(uint32_t*)src2);
        break;
    case 8:
        *(uint64_t*)dst = (*(uint64_t*)src1) & (*(uint64_t*)src2);
        break;
    default:
        assert(0 && "Unsupported data size for AND operation");
    }
}

void OrHandler::process_operation(ThreadContext* context, 
                                void* dst, void* src1, void* src2,
                                std::vector<Qualifier>& qualifiers) {
    // 实现OR指令的具体逻辑
    int bytes = TypeUtils::get_bytes(qualifiers);
    
    switch (bytes) {
    case 1:
        *(uint8_t*)dst = (*(uint8_t*)src1) | (*(uint8_t*)src2);
        break;
    case 2:
        *(uint16_t*)dst = (*(uint16_t*)src1) | (*(uint16_t*)src2);
        break;
    case 4:
        *(uint32_t*)dst = (*(uint32_t*)src1) | (*(uint32_t*)src2);
        break;
    case 8:
        *(uint64_t*)dst = (*(uint64_t*)src1) | (*(uint64_t*)src2);
        break;
    default:
        assert(0 && "Unsupported data size for OR operation");
    }
}

void XorHandler::process_operation(ThreadContext* context, 
                                 void* dst, void* src1, void* src2,
                                 std::vector<Qualifier>& qualifiers) {
    // 实现XOR指令的具体逻辑
    int bytes = TypeUtils::get_bytes(qualifiers);
    
    switch (bytes) {
    case 1:
        *(uint8_t*)dst = (*(uint8_t*)src1) ^ (*(uint8_t*)src2);
        break;
    case 2:
        *(uint16_t*)dst = (*(uint16_t*)src1) ^ (*(uint16_t*)src2);
        break;
    case 4:
        *(uint32_t*)dst = (*(uint32_t*)src1) ^ (*(uint32_t*)src2);
        break;
    case 8:
        *(uint64_t*)dst = (*(uint64_t*)src1) ^ (*(uint64_t*)src2);
        break;
    default:
        assert(0 && "Unsupported data size for XOR operation");
    }
}

void ShlHandler::process_operation(ThreadContext* context, 
                                 void* dst, void* src1, void* src2,
                                 std::vector<Qualifier>& qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    
    // 根据数据类型执行左移操作
    switch (bytes) {
    case 1: {
        *(uint8_t*)dst = (*(uint8_t*)src1) << (*(uint8_t*)src2);
        break;
    }
    case 2: {
        *(uint16_t*)dst = (*(uint16_t*)src1) << (*(uint16_t*)src2);
        break;
    }
    case 4: {
        *(uint32_t*)dst = (*(uint32_t*)src1) << (*(uint32_t*)src2);
        break;
    }
    case 8: {
        *(uint64_t*)dst = (*(uint64_t*)src1) << (*(uint64_t*)src2);
        break;
    }
    default:
        // 不支持的数据大小
        assert(0 && "Unsupported data size for SHL instruction");
    }
}

void ShrHandler::process_operation(ThreadContext* context, 
                                 void* dst, void* src1, void* src2,
                                 std::vector<Qualifier>& qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    
    // 根据数据类型执行逻辑右移操作（使用无符号类型）
    switch (bytes) {
    case 1: {
        *(uint8_t*)dst = (*(uint8_t*)src1) >> (*(uint8_t*)src2);
        break;
    }
    case 2: {
        *(uint16_t*)dst = (*(uint16_t*)src1) >> (*(uint16_t*)src2);
        break;
    }
    case 4: {
        *(uint32_t*)dst = (*(uint32_t*)src1) >> (*(uint32_t*)src2);
        break;
    }
    case 8: {
        *(uint64_t*)dst = (*(uint64_t*)src1) >> (*(uint64_t*)src2);
        break;
    }
    default:
        // 不支持的数据大小
        assert(0 && "Unsupported data size for SHR instruction");
    }
}