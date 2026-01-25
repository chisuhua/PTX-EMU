#include "ptxsim/register_analyzer.h"
#include "ptxsim/utils/qualifier_utils.h"
#include <iostream>

std::vector<RegisterInfo> RegisterAnalyzer::analyze_registers(
    const std::vector<StatementContext> &statements) {
    std::unordered_set<RegisterInfo, RegisterInfoHash> all_registers;

    for (const auto &stmt : statements) {
        extract_registers_from_statement(stmt, all_registers);
    }

    // 将unordered_set转换为vector返回
    std::vector<RegisterInfo> result(all_registers.begin(),
                                     all_registers.end());
    return result;
}

void RegisterAnalyzer::extract_registers_from_statement(
    const StatementContext &stmt,
    std::unordered_set<RegisterInfo, RegisterInfoHash> &registers) {
    // 首先处理寄存器声明语句
    if (stmt.statementType == S_REG) {
        auto *reg_stmt = static_cast<StatementContext::REG *>(stmt.statement);
        if (reg_stmt) {
            // 对于寄存器声明语句，提取寄存器信息
            if (reg_stmt->regNum == -1) {
                std::string reg_name = reg_stmt->regName;
                size_t reg_size = getBytes(reg_stmt->regDataType);
                if (reg_size == 0) {
                    // 如果无法确定大小，使用默认大小
                    reg_size = sizeof(uint32_t);
                }
                registers.insert(RegisterInfo(reg_name, -1, reg_size));
            } else {
                for (int i = 0; i < reg_stmt->regNum; ++i) {
                    std::string reg_name = reg_stmt->regName;
                    size_t reg_size = getBytes(reg_stmt->regDataType);
                    if (reg_size == 0) {
                        // 如果无法确定大小，使用默认大小
                        reg_size = sizeof(uint32_t);
                    }
                    registers.insert(RegisterInfo(reg_name, i, reg_size));
                }
            }
        }
    }

    // 然后处理所有语句中的操作数，提取实际使用的寄存器
    // extract_registers_from_all_operands(stmt, registers);
}

void RegisterAnalyzer::extract_registers_from_all_operands(
    const StatementContext &stmt,
    std::unordered_set<RegisterInfo, RegisterInfoHash> &registers) {
    // 根据语句类型获取操作数数组和数量
    void *stmt_ptr = stmt.statement;
    if (!stmt_ptr)
        return;

    // 通用处理：遍历所有可能的操作数
    switch (stmt.statementType) {

#define REG_EXTRACT_BRANCH(enum_val, type_name, opcount)
#define REG_EXTRACT_BARRIER(enum_val, type_name, opcount)
#define REG_EXTRACT_VOID_INSTR(enum_val, type_name, opcount)
#define REG_EXTRACT_PREDICATE_PREFIX(enum_val, type_name, opcount)
#define REG_EXTRACT_SIMPLE_STRING(enum_val, type_name, opcount)
#define REG_EXTRACT_SIMPLE_NAME(enum_val, type_name, opcount)
#define REG_EXTRACT_OPERAND_REG(enum_val, type_name, opcount)
#define REG_EXTRACT_OPERAND_CONST(enum_val, type_name, opcount)
#define REG_EXTRACT_OPERAND_MEMORY(enum_val, type_name, opcount)

#define REG_EXTRACT_ATOM_INSTR(enum_val, type_name, opcount)
#define REG_EXTRACT_WMMA_INSTR(enum_val, type_name, opcount)

#define REG_EXTRACT_GENERIC_INSTR(enum_val, type_name, opcount)                \
    case enum_val: {                                                           \
        if (auto *typed_stmt =                                                 \
                static_cast<StatementContext::type_name *>(stmt_ptr)) {        \
            for (int i = 0; i < opcount; ++i) {                                \
                extract_registers_from_operand(typed_stmt->operands[i],        \
                                               typed_stmt->qualifier,          \
                                               registers);                     \
            }                                                                  \
        }                                                                      \
        break;                                                                 \
    }
#define X(enum_val, type_name, str, opcount, struct_kind)                      \
    REG_EXTRACT_##struct_kind(enum_val, type_name, opcount)

#include "ptx_ir/ptx_op.def"
#undef X
    default:
        break;
    }
}

void RegisterAnalyzer::extract_registers_from_operand(
    const OperandContext &op, const std::vector<Qualifier> &qualifiers,
    std::unordered_set<RegisterInfo, RegisterInfoHash> &registers) {
    if (op.operandType == O_REG) {
        auto *reg = static_cast<OperandContext::REG *>(op.operand);
        if (reg) {
            // 获取寄存器的大小，根据qualifier确定
            size_t reg_size = getBytes(qualifiers);
            if (reg_size == 0) {
                // 如果没有明确的大小，使用默认的32位大小
                reg_size = sizeof(uint32_t);
            }
            // 添加寄存器到集合中，使用完整名称
            std::string full_name =
                reg->regName; // + std::to_string(reg->regIdx);
            registers.insert(RegisterInfo(full_name, reg->regIdx, reg_size));
        }
    } else if (op.operandType == O_VEC) {
        // 对于向量操作数，递归处理其中的每个元素
        auto *vec = static_cast<OperandContext::VEC *>(op.operand);
        if (vec) {
            for (const auto &element : vec->vec) {
                extract_registers_from_operand(element, qualifiers, registers);
            }
        }
    }
    // 其他类型如O_VAR, O_IMM, O_FA, O_PRED不需要分配寄存器空间
}