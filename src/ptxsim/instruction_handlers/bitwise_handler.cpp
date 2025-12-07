#include "ptxsim/instruction_handlers/bitwise_handler.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include "ptx_ir/ptx_types.h"
#include <cassert>
#include <vector>

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
    
    // 执行AND操作
    process_operation(context, to, op1, op2, ss->andQualifier);
}

void OrHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::OR*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->orOp[0], ss->orQualifier);
    void *op1 = context->get_operand_addr(ss->orOp[1], ss->orQualifier);
    void *op2 = context->get_operand_addr(ss->orOp[2], ss->orQualifier);
    
    // 执行OR操作
    process_operation(context, to, op1, op2, ss->orQualifier);
}

void XorHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::XOR*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->xorOp[0], ss->xorQualifier);
    void *op1 = context->get_operand_addr(ss->xorOp[1], ss->xorQualifier);
    void *op2 = context->get_operand_addr(ss->xorOp[2], ss->xorQualifier);
    
    // 执行XOR操作
    process_operation(context, to, op1, op2, ss->xorQualifier);
}

void ShlHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::SHL*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->shlOp[0], ss->shlQualifier);
    void *op1 = context->get_operand_addr(ss->shlOp[1], ss->shlQualifier);
    
    // 对于SHL，第三个操作数始终是U32类型
    std::vector<Qualifier> tq;
    tq.push_back(Qualifier::Q_U32);
    void *op2 = context->get_operand_addr(ss->shlOp[2], tq);
    
    // 执行SHL操作
    process_operation(context, to, op1, op2, ss->shlQualifier);
}

void ShrHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::SHR*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->shrOp[0], ss->shrQualifier);
    void *op1 = context->get_operand_addr(ss->shrOp[1], ss->shrQualifier);
    
    // 对于SHR，第三个操作数始终是U32类型
    std::vector<Qualifier> tq;
    tq.push_back(Qualifier::Q_U32);
    void *op2 = context->get_operand_addr(ss->shrOp[2], tq);
    
    // 执行SHR操作
    process_operation(context, to, op1, op2, ss->shrQualifier);
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
    // 实现SHL指令的具体逻辑
    int bytes = TypeUtils::get_bytes(qualifiers);
    
    switch (bytes) {
    case 1:
        *(uint8_t*)dst = (*(uint8_t*)src1) << (*(uint32_t*)src2);
        break;
    case 2:
        *(uint16_t*)dst = (*(uint16_t*)src1) << (*(uint32_t*)src2);
        break;
    case 4:
        *(uint32_t*)dst = (*(uint32_t*)src1) << (*(uint32_t*)src2);
        break;
    case 8:
        *(uint64_t*)dst = (*(uint64_t*)src1) << (*(uint32_t*)src2);
        break;
    default:
        assert(0 && "Unsupported data size for SHL operation");
    }
}

void ShrHandler::process_operation(ThreadContext* context, 
                                 void* dst, void* src1, void* src2,
                                 std::vector<Qualifier>& qualifiers) {
    // 实现SHR指令的具体逻辑
    int bytes = TypeUtils::get_bytes(qualifiers);
    
    switch (bytes) {
    case 1:
        *(uint8_t*)dst = (*(uint8_t*)src1) >> (*(uint32_t*)src2);
        break;
    case 2:
        *(uint16_t*)dst = (*(uint16_t*)src1) >> (*(uint32_t*)src2);
        break;
    case 4:
        *(uint32_t*)dst = (*(uint32_t*)src1) >> (*(uint32_t*)src2);
        break;
    case 8:
        *(uint64_t*)dst = (*(uint64_t*)src1) >> (*(uint32_t*)src2);
        break;
    default:
        assert(0 && "Unsupported data size for SHR operation");
    }
}