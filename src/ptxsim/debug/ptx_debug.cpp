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
        HANDLE_OP_CASE_2(MOV, op)
        HANDLE_OP_CASE_2(ST, op)
        HANDLE_OP_CASE_2(LD, op)
        HANDLE_OP_CASE_3(SETP, op)
        HANDLE_OP_CASE_3(MUL, op)
        HANDLE_OP_CASE_3(ADD, op)
        HANDLE_OP_CASE_3(SUB, op)
        HANDLE_OP_CASE_3(DIV, op)
        HANDLE_OP_CASE_2(ABS, op)
        HANDLE_OP_CASE_2(NEG, op)
        HANDLE_OP_CASE_3(MIN, op)
        HANDLE_OP_CASE_3(MAX, op)
        HANDLE_OP_CASE_3(AND, op)
        HANDLE_OP_CASE_3(OR, op)
        HANDLE_OP_CASE_3(XOR, op)
        HANDLE_OP_CASE_2(NOT, op)
        HANDLE_OP_CASE_3(SHL, op)
        HANDLE_OP_CASE_3(SHR, op)
        HANDLE_OP_CASE_1_STR(BRA, braTarget) // BRA只需要一个标签操作数
        HANDLE_OP_CASE_0(RET)                // RET没有操作数
        HANDLE_OP_CASE_1_STR(BAR, barType)   // BAR只需要类型操作数
        HANDLE_OP_CASE_2(RCP, op)
        HANDLE_OP_CASE_2(CVTA, op)
        HANDLE_OP_CASE_2(CVT, op)
        HANDLE_OP_CASE_4(SELP, op)
        HANDLE_OP_CASE_4(MAD, op)
        HANDLE_OP_CASE_4(FMA, op)
        HANDLE_OP_CASE_4(WMMA, op)
        HANDLE_OP_CASE_2(SQRT, op)
        HANDLE_OP_CASE_2(COS, op)
        HANDLE_OP_CASE_2(LG2, op)
        HANDLE_OP_CASE_2(EX2, op)
        HANDLE_OP_CASE_4(ATOM, op)
        HANDLE_OP_CASE_2(SIN, op)
        HANDLE_OP_CASE_2(RSQRT, op)
        HANDLE_OP_CASE_3(REM, op)
    default:
        // 对于其他未处理的指令类型，暂时使用占位符
        operands = "...";
        break;
    }

    return S2s(statement.statementType) + " " + operands;
}