#include "ptxsim/instruction_handlers/math_handler.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include <cassert>
#include <cmath>

void SQRT::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::SQRT *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->sqrtOp[0], ss->sqrtQualifier);
    void *op = context->get_operand_addr(ss->sqrtOp[1], ss->sqrtQualifier);

    // 执行SQRT操作
    int bytes = TypeUtils::get_bytes(ss->sqrtQualifier);
    bool is_float = TypeUtils::is_float_type(ss->sqrtQualifier);

    if (is_float) {
        if (bytes == 4) {
            *(float *)to = std::sqrt(*(float *)op);
        } else if (bytes == 8) {
            *(double *)to = std::sqrt(*(double *)op);
        }
    } else {
        // 整数开方
        switch (bytes) {
        case 1:
            *(uint8_t *)to = (uint8_t)std::sqrt(*(uint8_t *)op);
            break;
        case 2:
            *(uint16_t *)to = (uint16_t)std::sqrt(*(uint16_t *)op);
            break;
        case 4:
            *(uint32_t *)to = (uint32_t)std::sqrt(*(uint32_t *)op);
            break;
        case 8:
            *(uint64_t *)to = (uint64_t)std::sqrt(*(uint64_t *)op);
            break;
        default:
            assert(0 && "Unsupported data size for SQRT operation");
        }
    }
}

void SIN::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::SIN *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->sinOp[0], ss->sinQualifier);
    void *op = context->get_operand_addr(ss->sinOp[1], ss->sinQualifier);

    // 执行SIN操作
    int bytes = TypeUtils::get_bytes(ss->sinQualifier);
    bool is_float = TypeUtils::is_float_type(ss->sinQualifier);

    if (is_float) {
        if (bytes == 4) {
            *(float *)to = std::sin(*(float *)op);
        } else if (bytes == 8) {
            *(double *)to = std::sin(*(double *)op);
        }
    } else {
        // 整数正弦
        switch (bytes) {
        case 1:
            *(uint8_t *)to = (uint8_t)std::sin(*(uint8_t *)op);
            break;
        case 2:
            *(uint16_t *)to = (uint16_t)std::sin(*(uint16_t *)op);
            break;
        case 4:
            *(uint32_t *)to = (uint32_t)std::sin(*(uint32_t *)op);
            break;
        case 8:
            *(uint64_t *)to = (uint64_t)std::sin(*(uint64_t *)op);
            break;
        default:
            assert(0 && "Unsupported data size for SIN operation");
        }
    }
}

void COS::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::COS *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->cosOp[0], ss->cosQualifier);
    void *op = context->get_operand_addr(ss->cosOp[1], ss->cosQualifier);

    // 执行COS操作
    int bytes = TypeUtils::get_bytes(ss->cosQualifier);
    bool is_float = TypeUtils::is_float_type(ss->cosQualifier);

    if (is_float) {
        if (bytes == 4) {
            *(float *)to = std::cos(*(float *)op);
        } else if (bytes == 8) {
            *(double *)to = std::cos(*(double *)op);
        }
    } else {
        // 整数余弦
        switch (bytes) {
        case 1:
            *(uint8_t *)to = (uint8_t)std::cos(*(uint8_t *)op);
            break;
        case 2:
            *(uint16_t *)to = (uint16_t)std::cos(*(uint16_t *)op);
            break;
        case 4:
            *(uint32_t *)to = (uint32_t)std::cos(*(uint32_t *)op);
            break;
        case 8:
            *(uint64_t *)to = (uint64_t)std::cos(*(uint64_t *)op);
            break;
        default:
            assert(0 && "Unsupported data size for COS operation");
        }
    }
}