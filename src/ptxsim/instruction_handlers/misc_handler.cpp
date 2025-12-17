#include "ptxsim/instruction_handlers/misc_handler.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/instruction_handlers/arithmetic_handler.h"
#include "ptxsim/ptx_debug.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <algorithm>
#include <cassert>
#include <cmath>

void MovHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::MOV *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->movOp[0], ss->movQualifier);
    void *from = context->get_operand_addr(ss->movOp[1], ss->movQualifier);

    // 使用 PROCESS_OPERATION_2 宏执行 MOV
    // 操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_2(context, to, from, ss->movQualifier,
                        (OperandContext::REG *)ss->movOp[0].operand);
}

void MovHandler::process_operation(ThreadContext *context, void *dst, void *src,
                                   std::vector<Qualifier> &qualifiers) {
    // 执行MOV操作
    context->mov(src, dst, qualifiers);
}

void SetpHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::SETP *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->setpOp[0], ss->setpQualifier);
    void *op1 = context->get_operand_addr(ss->setpOp[1], ss->setpQualifier);
    void *op2 = context->get_operand_addr(ss->setpOp[2], ss->setpQualifier);

    // 执行SETP操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->setpQualifier,
                        (OperandContext::REG *)ss->setpOp[0].operand);
}

void SetpHandler::process_operation(ThreadContext *context, void *dst,
                                    void *src1, void *src2,
                                    std::vector<Qualifier> &qualifiers) {
    // 获取比较操作符
    Qualifier cmpOp = getCmpOpQualifier(qualifiers);
    Qualifier dtype = getDataQualifier(qualifiers);
    uint8_t result;

    SET_P_COMPARE(cmpOp, dtype, &result, src1, src2);

    *static_cast<uint8_t *>(dst) = result;
}

void AbsHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::ABS *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->absOp[0], ss->absQualifier);
    void *op = context->get_operand_addr(ss->absOp[1], ss->absQualifier);

    // 执行ABS操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_2(context, to, op, ss->absQualifier,
                        (OperandContext::REG *)ss->absOp[0].operand);
}

void AbsHandler::process_operation(ThreadContext *context, void *dst, void *src,
                                   std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // 根据数据类型执行绝对值操作
    switch (bytes) {
    case 1: {
        if (is_float) {
            *(uint8_t *)dst = std::abs(*(int8_t *)src);
        } else {
            // 对于无符号类型，绝对值就是本身
            *(uint8_t *)dst = *(uint8_t *)src;
        }
        break;
    }
    case 2: {
        if (is_float) {
            *(int16_t *)dst = std::abs(*(int16_t *)src);
        } else {
            // 对于无符号类型，绝对值就是本身
            *(uint16_t *)dst = *(uint16_t *)src;
        }
        break;
    }
    case 4: {
        if (is_float) {
            float val = *(float *)src;
            *(float *)dst = std::abs(val);
        } else {
            // 对于无符号类型，绝对值就是本身
            *(uint32_t *)dst = *(uint32_t *)src;
        }
        break;
    }
    case 8: {
        if (is_float) {
            double val = *(double *)src;
            *(double *)dst = std::abs(val);
        } else {
            // 对于无符号类型，绝对值就是本身
            *(uint64_t *)dst = *(uint64_t *)src;
        }
        break;
    }
    default:
        assert(0 && "Unsupported data size for ABS instruction");
    }
}

void MinHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::MIN *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->minOp[0], ss->minQualifier);
    void *op1 = context->get_operand_addr(ss->minOp[1], ss->minQualifier);
    void *op2 = context->get_operand_addr(ss->minOp[2], ss->minQualifier);

    // 执行MIN操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->minQualifier,
                        (OperandContext::REG *)ss->minOp[0].operand);
}

void MinHandler::process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2,
                                   std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // 根据数据类型执行最小值操作
    switch (bytes) {
    case 1: {
        if (is_float) {
            *(uint8_t *)dst = std::min(*(uint8_t *)src1, *(uint8_t *)src2);
        } else {
            *(uint8_t *)dst = std::min(*(uint8_t *)src1, *(uint8_t *)src2);
        }
        break;
    }
    case 2: {
        if (is_float) {
            *(uint16_t *)dst = std::min(*(uint16_t *)src1, *(uint16_t *)src2);
        } else {
            *(uint16_t *)dst = std::min(*(uint16_t *)src1, *(uint16_t *)src2);
        }
        break;
    }
    case 4: {
        if (is_float) {
            *(float *)dst = std::min(*(float *)src1, *(float *)src2);
        } else {
            *(uint32_t *)dst = std::min(*(uint32_t *)src1, *(uint32_t *)src2);
        }
        break;
    }
    case 8: {
        if (is_float) {
            *(double *)dst = std::min(*(double *)src1, *(double *)src2);
        } else {
            *(uint64_t *)dst = std::min(*(uint64_t *)src1, *(uint64_t *)src2);
        }
        break;
    }
    default:
        assert(0 && "Unsupported data size for MIN instruction");
    }
}

void MaxHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::MAX *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->maxOp[0], ss->maxQualifier);
    void *op1 = context->get_operand_addr(ss->maxOp[1], ss->maxQualifier);
    void *op2 = context->get_operand_addr(ss->maxOp[2], ss->maxQualifier);

    // 执行MAX操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_3(context, to, op1, op2, ss->maxQualifier,
                        (OperandContext::REG *)ss->maxOp[0].operand);
}

void MaxHandler::process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2,
                                   std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // 根据数据类型执行最大值操作
    switch (bytes) {
    case 1: {
        if (is_float) {
            *(uint8_t *)dst = std::max(*(uint8_t *)src1, *(uint8_t *)src2);
        } else {
            *(uint8_t *)dst = std::max(*(uint8_t *)src1, *(uint8_t *)src2);
        }
        break;
    }
    case 2: {
        if (is_float) {
            *(uint16_t *)dst = std::max(*(uint16_t *)src1, *(uint16_t *)src2);
        } else {
            *(uint16_t *)dst = std::max(*(uint16_t *)src1, *(uint16_t *)src2);
        }
        break;
    }
    case 4: {
        if (is_float) {
            *(float *)dst = std::max(*(float *)src1, *(float *)src2);
        } else {
            *(uint32_t *)dst = std::max(*(uint32_t *)src1, *(uint32_t *)src2);
        }
        break;
    }
    case 8: {
        if (is_float) {
            *(double *)dst = std::max(*(double *)src1, *(double *)src2);
        } else {
            *(uint64_t *)dst = std::max(*(uint64_t *)src1, *(uint64_t *)src2);
        }
        break;
    }
    default:
        assert(0 && "Unsupported data size for MAX instruction");
    }
}

void RcpHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::RCP *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->rcpOp[0], ss->rcpQualifier);
    void *op = context->get_operand_addr(ss->rcpOp[1], ss->rcpQualifier);

    // 执行RCP操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_2(context, to, op, ss->rcpQualifier,
                        (OperandContext::REG *)ss->rcpOp[0].operand);
}

void RcpHandler::process_operation(ThreadContext *context, void *dst, void *src,
                                   std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // RCP指令只支持浮点类型
    assert(is_float && "RCP instruction only supports floating point types");

    // 根据数据类型执行倒数操作
    switch (bytes) {
    case 4: {
        float val = *(float *)src;
        *(float *)dst = 1.0f / val;
        break;
    }
    case 8: {
        double val = *(double *)src;
        *(double *)dst = 1.0 / val;
        break;
    }
    default:
        assert(0 && "Unsupported data size for RCP instruction");
    }
}

void NegHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::NEG *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->negOp[0], ss->negQualifier);
    void *op = context->get_operand_addr(ss->negOp[1], ss->negQualifier);

    // 执行NEG操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_2(context, to, op, ss->negQualifier,
                        (OperandContext::REG *)ss->negOp[0].operand);
}

void NegHandler::process_operation(ThreadContext *context, void *dst, void *src,
                                   std::vector<Qualifier> &qualifiers) {
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // 根据数据类型执行取负操作
    switch (bytes) {
    case 2: {
        if (is_float) {
            *(int16_t *)dst = -(*(int16_t *)src);
        } else {
            *(uint16_t *)dst = -(uint16_t)(*(uint16_t *)src);
        }
        break;
    }
    case 4: {
        if (is_float) {
            float val = *(float *)src;
            *(float *)dst = -val;
        } else {
            *(uint32_t *)dst = -(uint32_t)(*(uint32_t *)src);
        }
        break;
    }
    case 8: {
        if (is_float) {
            double val = *(double *)src;
            *(double *)dst = -val;
        } else {
            *(uint64_t *)dst = -(uint64_t)(*(uint64_t *)src);
        }
        break;
    }
    default:
        assert(0 && "Unsupported data size for NEG instruction");
    }
}
