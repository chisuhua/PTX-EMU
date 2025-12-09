#include "ptxsim/ptx_debug.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/ptx_types.h"

// 定义操作数处理宏
#define HANDLE_OP_CASE_0(type) \
    case StatementType::S_##type: { \
        operands = ""; \
        break; \
    }

#define HANDLE_OP_CASE_1(type, op_field) \
    case StatementType::S_##type: { \
        auto *stmt = static_cast<StatementContext::type *>(statement.statement); \
        if (stmt) { \
            operands = stmt->op_field[0].toString(); \
        } \
        break; \
    }

#define HANDLE_OP_CASE_1_STR(type, op_field) \
    case StatementType::S_##type: { \
        auto *stmt = static_cast<StatementContext::type *>(statement.statement); \
        if (stmt) { \
            operands = stmt->op_field; \
        } \
        break; \
    }

#define HANDLE_OP_CASE_2(type, op_field) \
    case StatementType::S_##type: { \
        auto *stmt = static_cast<StatementContext::type *>(statement.statement); \
        if (stmt) { \
            operands = stmt->op_field[0].toString() + ", " + \
                       stmt->op_field[1].toString(); \
        } \
        break; \
    }

#define HANDLE_OP_CASE_3(type, op_field) \
    case StatementType::S_##type: { \
        auto *stmt = static_cast<StatementContext::type *>(statement.statement); \
        if (stmt) { \
            operands = stmt->op_field[0].toString() + ", " + \
                       stmt->op_field[1].toString() + ", " + \
                       stmt->op_field[2].toString(); \
        } \
        break; \
    }

#define HANDLE_OP_CASE_4(type, op_field) \
    case StatementType::S_##type: { \
        auto *stmt = static_cast<StatementContext::type *>(statement.statement); \
        if (stmt) { \
            operands = stmt->op_field[0].toString() + ", " + \
                       stmt->op_field[1].toString() + ", " + \
                       stmt->op_field[2].toString() + ", " + \
                       stmt->op_field[3].toString(); \
        } \
        break; \
    }

std::string ptxsim::DebugConfig::get_full_instruction_string(const StatementContext& statement) {
    std::string operands = "";
    
    // 针对不同类型的操作数进行处理
    switch (statement.statementType) {
        HANDLE_OP_CASE_2(MOV, movOp)
        HANDLE_OP_CASE_2(ST, stOp)
        HANDLE_OP_CASE_2(LD, ldOp)
        HANDLE_OP_CASE_3(SETP, setpOp)
        HANDLE_OP_CASE_3(MUL, mulOp)
        HANDLE_OP_CASE_3(ADD, addOp)
        HANDLE_OP_CASE_3(SUB, subOp)
        HANDLE_OP_CASE_3(DIV, divOp)
        HANDLE_OP_CASE_2(ABS, absOp)
        HANDLE_OP_CASE_2(NEG, negOp)
        HANDLE_OP_CASE_3(MIN, minOp)
        HANDLE_OP_CASE_3(MAX, maxOp)
        HANDLE_OP_CASE_3(AND, andOp)
        HANDLE_OP_CASE_3(OR, orOp)
        HANDLE_OP_CASE_3(XOR, xorOp)
        HANDLE_OP_CASE_2(NOT, notOp)
        HANDLE_OP_CASE_3(SHL, shlOp)
        HANDLE_OP_CASE_3(SHR, shrOp)
        HANDLE_OP_CASE_1_STR(BRA, braTarget)  // BRA只需要一个标签操作数
        HANDLE_OP_CASE_0(RET)                 // RET没有操作数
        HANDLE_OP_CASE_1_STR(BAR, barType)    // BAR只需要类型操作数
        default:
            // 对于其他未处理的指令类型，暂时使用占位符
            operands = "...";
            break;
    }
    
    return S2s(statement.statementType) + " " + operands;
}