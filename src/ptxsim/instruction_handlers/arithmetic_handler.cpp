#include "ptxsim/instruction_handlers/arithmetic_handler.h"
#include "ptxsim/instruction_processor_utils.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

void AddHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::ADD *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->addOp[0], ss->addQualifier);
    void *op1 = context->get_operand_addr(ss->addOp[1], ss->addQualifier);
    void *op2 = context->get_operand_addr(ss->addOp[2], ss->addQualifier);

    // 执行ADD操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->addQualifier,
                        (OperandContext::REG *)ss->addOp[0].operand);
}

void AddHandler::process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2,
                                   std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // 根据数据类型执行加法操作
    switch (bytes) {
    case 1: {
        if (is_float) {
            *(uint8_t *)dst = (*(uint8_t *)src1) + (*(uint8_t *)src2);
        } else {
            *(uint8_t *)dst = (*(uint8_t *)src1) + (*(uint8_t *)src2);
        }
        break;
    }
    case 2: {
        if (is_float) {
            *(uint16_t *)dst = (*(uint16_t *)src1) + (*(uint16_t *)src2);
        } else {
            *(uint16_t *)dst = (*(uint16_t *)src1) + (*(uint16_t *)src2);
        }
        break;
    }
    case 4: {
        if (is_float) {
            *(float *)dst = (*(float *)src1) + (*(float *)src2);
        } else {
            *(uint32_t *)dst = (*(uint32_t *)src1) + (*(uint32_t *)src2);
        }
        break;
    }
    case 8: {
        if (is_float) {
            *(double *)dst = (*(double *)src1) + (*(double *)src2);
        } else {
            *(uint64_t *)dst = (*(uint64_t *)src1) + (*(uint64_t *)src2);
        }
        break;
    }
    default:
        // 不支持的数据大小
        assert(0 && "Unsupported data size for ADD instruction");
    }
}

void SubHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::SUB *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->subOp[0], ss->subQualifier);
    void *op1 = context->get_operand_addr(ss->subOp[1], ss->subQualifier);
    void *op2 = context->get_operand_addr(ss->subOp[2], ss->subQualifier);

    // 执行SUB操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->subQualifier,
                        (OperandContext::REG *)ss->subOp[0].operand);
}

void SubHandler::process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2,
                                   std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // 根据数据类型执行减法操作
    switch (bytes) {
    case 1: {
        if (is_float) {
            *(uint8_t *)dst = (*(uint8_t *)src1) - (*(uint8_t *)src2);
        } else {
            *(uint8_t *)dst = (*(uint8_t *)src1) - (*(uint8_t *)src2);
        }
        break;
    }
    case 2: {
        if (is_float) {
            *(uint16_t *)dst = (*(uint16_t *)src1) - (*(uint16_t *)src2);
        } else {
            *(uint16_t *)dst = (*(uint16_t *)src1) - (*(uint16_t *)src2);
        }
        break;
    }
    case 4: {
        if (is_float) {
            *(float *)dst = (*(float *)src1) - (*(float *)src2);
        } else {
            *(uint32_t *)dst = (*(uint32_t *)src1) - (*(uint32_t *)src2);
        }
        break;
    }
    case 8: {
        if (is_float) {
            *(double *)dst = (*(double *)src1) - (*(double *)src2);
        } else {
            *(uint64_t *)dst = (*(uint64_t *)src1) - (*(uint64_t *)src2);
        }
        break;
    }
    default:
        // 不支持的数据大小
        assert(0 && "Unsupported data size for SUB instruction");
    }
}

void MulHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::MUL *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->mulOp[0], ss->mulQualifier);
    void *op1 = context->get_operand_addr(ss->mulOp[1], ss->mulQualifier);
    void *op2 = context->get_operand_addr(ss->mulOp[2], ss->mulQualifier);

    // 执行MUL操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->mulQualifier,
                        (OperandContext::REG *)ss->mulOp[0].operand);
}

void MulHandler::process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2,
                                   std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // 根据数据类型执行乘法操作
    switch (bytes) {
    case 1: {
        if (is_float) {
            *(uint8_t *)dst = (*(uint8_t *)src1) * (*(uint8_t *)src2);
        } else {
            *(uint8_t *)dst = (*(uint8_t *)src1) * (*(uint8_t *)src2);
        }
        break;
    }
    case 2: {
        if (is_float) {
            *(uint16_t *)dst = (*(uint16_t *)src1) * (*(uint16_t *)src2);
        } else {
            *(uint16_t *)dst = (*(uint16_t *)src1) * (*(uint16_t *)src2);
        }
        break;
    }
    case 4: {
        if (is_float) {
            *(float *)dst = (*(float *)src1) * (*(float *)src2);
        } else {
            *(uint32_t *)dst = (*(uint32_t *)src1) * (*(uint32_t *)src2);
        }
        break;
    }
    case 8: {
        if (is_float) {
            *(double *)dst = (*(double *)src1) * (*(double *)src2);
        } else {
            *(uint64_t *)dst = (*(uint64_t *)src1) * (*(uint64_t *)src2);
        }
        break;
    }
    default:
        // 不支持的数据大小
        assert(0 && "Unsupported data size for MUL instruction");
    }
}

void DivHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::DIV *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->divOp[0], ss->divQualifier);
    void *op1 = context->get_operand_addr(ss->divOp[1], ss->divQualifier);
    void *op2 = context->get_operand_addr(ss->divOp[2], ss->divQualifier);

    // 执行DIV操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->divQualifier,
                        (OperandContext::REG *)ss->divOp[0].operand);
}

void DivHandler::process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2,
                                   std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // 根据数据类型执行除法操作
    switch (bytes) {
    case 1: {
        if (is_float) {
            *(uint8_t *)dst = (*(uint8_t *)src1) / (*(uint8_t *)src2);
        } else {
            *(uint8_t *)dst = (*(uint8_t *)src1) / (*(uint8_t *)src2);
        }
        break;
    }
    case 2: {
        if (is_float) {
            *(uint16_t *)dst = (*(uint16_t *)src1) / (*(uint16_t *)src2);
        } else {
            *(uint16_t *)dst = (*(uint16_t *)src1) / (*(uint16_t *)src2);
        }
        break;
    }
    case 4: {
        if (is_float) {
            *(float *)dst = (*(float *)src1) / (*(float *)src2);
        } else {
            *(uint32_t *)dst = (*(uint32_t *)src1) / (*(uint32_t *)src2);
        }
        break;
    }
    case 8: {
        if (is_float) {
            *(double *)dst = (*(double *)src1) / (*(double *)src2);
        } else {
            *(uint64_t *)dst = (*(uint64_t *)src1) / (*(uint64_t *)src2);
        }
        break;
    }
    default:
        // 不支持的数据大小
        assert(0 && "Unsupported data size for DIV instruction");
    }
}

void MadHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::MAD *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->madOp[0], ss->madQualifier);
    void *op1 = context->get_operand_addr(ss->madOp[1], ss->madQualifier);
    void *op2 = context->get_operand_addr(ss->madOp[2], ss->madQualifier);
    void *op3 = context->get_operand_addr(ss->madOp[3], ss->madQualifier);

    // 执行MAD操作
    PROCESS_OPERATION_4(context, to, op1, op2, op3, ss->madQualifier,
                        (OperandContext::REG *)ss->madOp[0].operand);
}

void MadHandler::process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2, void *src3,
                                   std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // 根据数据类型执行乘加操作 (dst = src1 * src2 + src3)
    switch (bytes) {
    case 1: {
        if (is_float) {
            *(uint8_t *)dst =
                (*(uint8_t *)src1) * (*(uint8_t *)src2) + (*(uint8_t *)src3);
        } else {
            *(uint8_t *)dst =
                (*(uint8_t *)src1) * (*(uint8_t *)src2) + (*(uint8_t *)src3);
        }
        break;
    }
    case 2: {
        if (is_float) {
            *(uint16_t *)dst =
                (*(uint16_t *)src1) * (*(uint16_t *)src2) + (*(uint16_t *)src3);
        } else {
            *(uint16_t *)dst =
                (*(uint16_t *)src1) * (*(uint16_t *)src2) + (*(uint16_t *)src3);
        }
        break;
    }
    case 4: {
        if (is_float) {
            *(float *)dst =
                (*(float *)src1) * (*(float *)src2) + (*(float *)src3);
        } else {
            *(uint32_t *)dst =
                (*(uint32_t *)src1) * (*(uint32_t *)src2) + (*(uint32_t *)src3);
        }
        break;
    }
    case 8: {
        if (is_float) {
            *(double *)dst =
                (*(double *)src1) * (*(double *)src2) + (*(double *)src3);
        } else {
            *(uint64_t *)dst =
                (*(uint64_t *)src1) * (*(uint64_t *)src2) + (*(uint64_t *)src3);
        }
        break;
    }
    default:
        // 不支持的数据大小
        assert(0 && "Unsupported data size for MAD instruction");
    }
}

void FmaHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::FMA *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->fmaOp[0], ss->fmaQualifier);
    void *op1 = context->get_operand_addr(ss->fmaOp[1], ss->fmaQualifier);
    void *op2 = context->get_operand_addr(ss->fmaOp[2], ss->fmaQualifier);
    void *op3 = context->get_operand_addr(ss->fmaOp[3], ss->fmaQualifier);

    // 执行FMA操作
    PROCESS_OPERATION_4(context, to, op1, op2, op3, ss->fmaQualifier,
                        (OperandContext::REG *)ss->fmaOp[0].operand);
}

void FmaHandler::process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2, void *src3,
                                   std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // 根据数据类型执行融合乘加操作 (dst = src1 * src2 + src3)
    switch (bytes) {
    case 1: {
        if (is_float) {
            *(uint8_t *)dst =
                (*(uint8_t *)src1) * (*(uint8_t *)src2) + (*(uint8_t *)src3);
        } else {
            *(uint8_t *)dst =
                (*(uint8_t *)src1) * (*(uint8_t *)src2) + (*(uint8_t *)src3);
        }
        break;
    }
    case 2: {
        if (is_float) {
            *(uint16_t *)dst =
                (*(uint16_t *)src1) * (*(uint16_t *)src2) + (*(uint16_t *)src3);
        } else {
            *(uint16_t *)dst =
                (*(uint16_t *)src1) * (*(uint16_t *)src2) + (*(uint16_t *)src3);
        }
        break;
    }
    case 4: {
        if (is_float) {
            *(float *)dst =
                (*(float *)src1) * (*(float *)src2) + (*(float *)src3);
        } else {
            *(uint32_t *)dst =
                (*(uint32_t *)src1) * (*(uint32_t *)src2) + (*(uint32_t *)src3);
        }
        break;
    }
    case 8: {
        if (is_float) {
            *(double *)dst =
                (*(double *)src1) * (*(double *)src2) + (*(double *)src3);
        } else {
            *(uint64_t *)dst =
                (*(uint64_t *)src1) * (*(uint64_t *)src2) + (*(uint64_t *)src3);
        }
        break;
    }
    default:
        // 不支持的数据大小
        assert(0 && "Unsupported data size for FMA instruction");
    }
}
