#include "ptxsim/instruction_handlers/arithmetic_handler.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include <cassert>

void ArithmeticHandler::execute(ThreadContext* context, StatementContext& stmt) {
    // 这是一个抽象基类，不应该被直接调用
    assert(0 && "ArithmeticHandler::execute should not be called directly");
}

void AddHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::ADD*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->addOp[0], ss->addQualifier);
    void *op1 = context->get_operand_addr(ss->addOp[1], ss->addQualifier);
    void *op2 = context->get_operand_addr(ss->addOp[2], ss->addQualifier);
    
    // 执行加法操作
    process_operation(context, to, op1, op2, ss->addQualifier);
}

void SubHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::SUB*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->subOp[0], ss->subQualifier);
    void *op1 = context->get_operand_addr(ss->subOp[1], ss->subQualifier);
    void *op2 = context->get_operand_addr(ss->subOp[2], ss->subQualifier);
    
    // 执行减法操作
    process_operation(context, to, op1, op2, ss->subQualifier);
}

void MulHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::MUL*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->mulOp[0], ss->mulQualifier);
    void *op1 = context->get_operand_addr(ss->mulOp[1], ss->mulQualifier);
    void *op2 = context->get_operand_addr(ss->mulOp[2], ss->mulQualifier);
    
    // 执行乘法操作
    process_operation(context, to, op1, op2, ss->mulQualifier);
}

void DivHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::DIV*)stmt.statement;
    
    // 获取操作数地址
    void *to = context->get_operand_addr(ss->divOp[0], ss->divQualifier);
    void *op1 = context->get_operand_addr(ss->divOp[1], ss->divQualifier);
    void *op2 = context->get_operand_addr(ss->divOp[2], ss->divQualifier);
    
    // 检查除零错误
    int bytes = TypeUtils::get_bytes(ss->divQualifier);
    bool is_float = TypeUtils::is_float_type(ss->divQualifier);
    
    bool zero_divisor = false;
    switch (bytes) {
    case 1:
        zero_divisor = (*(uint8_t*)op2) == 0;
        break;
    case 2:
        zero_divisor = (*(uint16_t*)op2) == 0;
        break;
    case 4:
        if (is_float) {
            zero_divisor = (*(float*)op2) == 0.0f;
        } else {
            zero_divisor = (*(uint32_t*)op2) == 0;
        }
        break;
    case 8:
        if (is_float) {
            zero_divisor = (*(double*)op2) == 0.0;
        } else {
            zero_divisor = (*(uint64_t*)op2) == 0;
        }
        break;
    }
    
    if (zero_divisor) {
        // 处理除零错误
        assert(0 && "Division by zero");
        return;
    }
    
    // 执行除法操作
    process_operation(context, to, op1, op2, ss->divQualifier);
}

void AddHandler::process_operation(ThreadContext* context, 
                                 void* dst, void* src1, void* src2,
                                 std::vector<Qualifier>& qualifiers) {
    // 实现ADD指令的具体逻辑
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    
    if (is_float) {
        if (bytes == 4) {
            *(float*)dst = (*(float*)src1) + (*(float*)src2);
        } else if (bytes == 8) {
            *(double*)dst = (*(double*)src1) + (*(double*)src2);
        }
    } else {
        // 整数加法
        switch (bytes) {
        case 1:
            *(uint8_t*)dst = (*(uint8_t*)src1) + (*(uint8_t*)src2);
            break;
        case 2:
            *(uint16_t*)dst = (*(uint16_t*)src1) + (*(uint16_t*)src2);
            break;
        case 4:
            *(uint32_t*)dst = (*(uint32_t*)src1) + (*(uint32_t*)src2);
            break;
        case 8:
            *(uint64_t*)dst = (*(uint64_t*)src1) + (*(uint64_t*)src2);
            break;
        default:
            assert(0 && "Unsupported data size for ADD operation");
        }
    }
}

void SubHandler::process_operation(ThreadContext* context, 
                                 void* dst, void* src1, void* src2,
                                 std::vector<Qualifier>& qualifiers) {
    // 实现SUB指令的具体逻辑
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    
    if (is_float) {
        if (bytes == 4) {
            *(float*)dst = (*(float*)src1) - (*(float*)src2);
        } else if (bytes == 8) {
            *(double*)dst = (*(double*)src1) - (*(double*)src2);
        }
    } else {
        // 整数减法
        switch (bytes) {
        case 1:
            *(uint8_t*)dst = (*(uint8_t*)src1) - (*(uint8_t*)src2);
            break;
        case 2:
            *(uint16_t*)dst = (*(uint16_t*)src1) - (*(uint16_t*)src2);
            break;
        case 4:
            *(uint32_t*)dst = (*(uint32_t*)src1) - (*(uint32_t*)src2);
            break;
        case 8:
            *(uint64_t*)dst = (*(uint64_t*)src1) - (*(uint64_t*)src2);
            break;
        default:
            assert(0 && "Unsupported data size for SUB operation");
        }
    }
}

void MulHandler::process_operation(ThreadContext* context, 
                                 void* dst, void* src1, void* src2,
                                 std::vector<Qualifier>& qualifiers) {
    // 实现MUL指令的具体逻辑
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    
    if (is_float) {
        if (bytes == 4) {
            *(float*)dst = (*(float*)src1) * (*(float*)src2);
        } else if (bytes == 8) {
            *(double*)dst = (*(double*)src1) * (*(double*)src2);
        }
    } else {
        // 整数乘法
        switch (bytes) {
        case 1:
            *(uint8_t*)dst = (*(uint8_t*)src1) * (*(uint8_t*)src2);
            break;
        case 2:
            *(uint16_t*)dst = (*(uint16_t*)src1) * (*(uint16_t*)src2);
            break;
        case 4:
            *(uint32_t*)dst = (*(uint32_t*)src1) * (*(uint32_t*)src2);
            break;
        case 8:
            *(uint64_t*)dst = (*(uint64_t*)src1) * (*(uint64_t*)src2);
            break;
        default:
            assert(0 && "Unsupported data size for MUL operation");
        }
    }
}

void DivHandler::process_operation(ThreadContext* context, 
                                 void* dst, void* src1, void* src2,
                                 std::vector<Qualifier>& qualifiers) {
    // 实现DIV指令的具体逻辑
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);
    
    if (is_float) {
        if (bytes == 4) {
            *(float*)dst = (*(float*)src1) / (*(float*)src2);
        } else if (bytes == 8) {
            *(double*)dst = (*(double*)src1) / (*(double*)src2);
        }
    } else {
        // 整数除法
        switch (bytes) {
        case 1:
            *(uint8_t*)dst = (*(uint8_t*)src1) / (*(uint8_t*)src2);
            break;
        case 2:
            *(uint16_t*)dst = (*(uint16_t*)src1) / (*(uint16_t*)src2);
            break;
        case 4:
            *(uint32_t*)dst = (*(uint32_t*)src1) / (*(uint32_t*)src2);
            break;
        case 8:
            *(uint64_t*)dst = (*(uint64_t*)src1) / (*(uint64_t*)src2);
            break;
        default:
            assert(0 && "Unsupported data size for DIV operation");
        }
    }
}