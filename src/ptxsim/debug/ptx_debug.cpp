#include "ptxsim/ptx_debug.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"

// 定义操作数处理宏
#define HANDLE_OP_CASE_0(type)                                                 \
    case StatementType::S_##type: {                                            \
        operands = "";                                                         \
        break;                                                                 \
    }

#define HANDLE_OP_CASE_1(type, op_field)                                       \
    case StatementType::S_##type: {                                            \
        auto *stmt =                                                           \
            static_cast<StatementContext::type *>(statement.statement);        \
        if (stmt) {                                                            \
            operands = stmt->op_field[0].toString();                           \
        }                                                                      \
        break;                                                                 \
    }

#define HANDLE_OP_CASE_1_STR(type, op_field)                                   \
    case StatementType::S_##type: {                                            \
        auto *stmt =                                                           \
            static_cast<StatementContext::type *>(statement.statement);        \
        if (stmt) {                                                            \
            operands = stmt->op_field;                                         \
        }                                                                      \
        break;                                                                 \
    }

#define HANDLE_OP_CASE_2(type, op_field)                                       \
    case StatementType::S_##type: {                                            \
        auto *stmt =                                                           \
            static_cast<StatementContext::type *>(statement.statement);        \
        if (stmt) {                                                            \
            operands = stmt->op_field[0].toString() + ", " +                   \
                       stmt->op_field[1].toString();                           \
        }                                                                      \
        break;                                                                 \
    }

#define HANDLE_OP_CASE_3(type, op_field)                                       \
    case StatementType::S_##type: {                                            \
        auto *stmt =                                                           \
            static_cast<StatementContext::type *>(statement.statement);        \
        if (stmt) {                                                            \
            operands = stmt->op_field[0].toString() + ", " +                   \
                       stmt->op_field[1].toString() + ", " +                   \
                       stmt->op_field[2].toString();                           \
        }                                                                      \
        break;                                                                 \
    }

#define HANDLE_OP_CASE_4(type, op_field)                                       \
    case StatementType::S_##type: {                                            \
        auto *stmt =                                                           \
            static_cast<StatementContext::type *>(statement.statement);        \
        if (stmt) {                                                            \
            operands = stmt->op_field[0].toString() + ", " +                   \
                       stmt->op_field[1].toString() + ", " +                   \
                       stmt->op_field[2].toString() + ", " +                   \
                       stmt->op_field[3].toString();                           \
        }                                                                      \
        break;                                                                 \
    }

std::string ptxsim::DebugConfig::get_full_instruction_string(
    const StatementContext &statement) {
    std::string operands = "";

    // 针对不同类型的操作数进行处理
    switch (statement.statementType) {
        HANDLE_OP_CASE_2(MOV, operands)
        HANDLE_OP_CASE_2(ST, operands)
        HANDLE_OP_CASE_2(LD, operands)
        HANDLE_OP_CASE_3(SETP, operands)
        HANDLE_OP_CASE_3(MUL, operands)
        HANDLE_OP_CASE_3(ADD, operands)
        HANDLE_OP_CASE_3(SUB, operands)
        HANDLE_OP_CASE_3(DIV, operands)
        HANDLE_OP_CASE_2(ABS, operands)
        HANDLE_OP_CASE_2(NEG, operands)
        HANDLE_OP_CASE_3(MIN, operands)
        HANDLE_OP_CASE_3(MAX, operands)
        HANDLE_OP_CASE_3(AND, operands)
        HANDLE_OP_CASE_3(OR, operands)
        HANDLE_OP_CASE_3(XOR, operands)
        HANDLE_OP_CASE_2(NOT, operands)
        HANDLE_OP_CASE_3(SHL, operands)
        HANDLE_OP_CASE_3(SHR, operands)
        HANDLE_OP_CASE_1_STR(BRA, braTarget) // BRA只需要一个标签操作数
        HANDLE_OP_CASE_0(RET)                // RET没有操作数
        HANDLE_OP_CASE_1_STR(BAR, barType)   // BAR只需要类型操作数
        HANDLE_OP_CASE_2(RCP, operands)
        HANDLE_OP_CASE_2(CVTA, operands)
        HANDLE_OP_CASE_2(CVT, operands)
        HANDLE_OP_CASE_4(SELP, operands)
        HANDLE_OP_CASE_4(MAD, operands)
        HANDLE_OP_CASE_4(FMA, operands)
        HANDLE_OP_CASE_4(WMMA, operands)
        HANDLE_OP_CASE_2(SQRT, operands)
        HANDLE_OP_CASE_2(COS, operands)
        HANDLE_OP_CASE_2(LG2, operands)
        HANDLE_OP_CASE_2(EX2, operands)
        HANDLE_OP_CASE_4(ATOM, operands)
        HANDLE_OP_CASE_2(SIN, operands)
        HANDLE_OP_CASE_2(RSQRT, operands)
        HANDLE_OP_CASE_3(REM, operands)
    default:
        // 对于其他未处理的指令类型，暂时使用占位符
        operands = "...";
        break;
    }

    return S2s(statement.statementType) + " " + operands;
}