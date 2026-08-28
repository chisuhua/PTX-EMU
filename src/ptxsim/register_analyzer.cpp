#include "ptxsim/register_analyzer.h"
#include "ptxsim/utils/qualifier_utils.h"
#include <iostream>

std::vector<RegisterInfo> RegisterAnalyzer::analyze_registers(
    const std::vector<ptxemu::ir::StatementContext> &statements) {
    std::unordered_set<RegisterInfo, RegisterInfoHash> all_registers;

    for (const auto &stmt : statements) {
        extract_registers_from_statement(stmt, all_registers);
    }

    // 将unordered_set转换为vector返回
    std::vector<RegisterInfo> result(all_registers.begin(),
                                     all_registers.end());
    return result;
}

std::vector<uint32_t> RegisterAnalyzer::get_dest_registers_as_ids(
    const ptxemu::ir::StatementContext &stmt) {
    std::vector<uint32_t> result;
    stmt.visit([&result](const auto &instr) {
        using T = std::decay_t<decltype(instr)>;
        if constexpr (requires { instr.operands; }) {
            if (!instr.operands.empty()) {
                const auto &dst = instr.operands[0];
                if (dst.kind() == OperandKind::REG) {
                    result.push_back(
                        std::get<RegOperand>(dst.data).index);
                }
                // VecOperand (tex/ld.v4) — Phase 8.B TODO
                // st/red/prefetch's operands[0] is AddrOperand → kind() != REG → skip
            }
        }
    });
    return result;
}

void RegisterAnalyzer::extract_registers_from_statement(
    const ptxemu::ir::StatementContext &stmt,
    std::unordered_set<RegisterInfo, RegisterInfoHash> &registers) {
    // 首先处理寄存器声明语句
    if (stmt.type == ptxemu::ir::StatementType::S_REG) {
        // auto *reg_stmt = static_cast<StatementContext::REG
        // *>(stmt.statement);
        const DeclarationInstr &reg_stmt =
            std::get<DeclarationInstr>(stmt.data);
        // 对于寄存器声明语句，提取寄存器信息
        if (reg_stmt.array_size == -1) {
            std::string reg_name = reg_stmt.name;
            size_t reg_size = Q2bytes(reg_stmt.dataType);
            if (reg_size == 0) {
                // 如果无法确定大小，使用默认大小
                reg_size = sizeof(uint32_t);
            }
            registers.insert(RegisterInfo(reg_name, -1, reg_size));
        } else {
            for (int i = 0; i < reg_stmt.array_size; ++i) {
                std::string reg_name = reg_stmt.name;
                size_t reg_size = Q2bytes(reg_stmt.dataType);
                if (reg_size == 0) {
                    // 如果无法确定大小，使用默认大小
                    reg_size = sizeof(uint32_t);
                }
                registers.insert(RegisterInfo(reg_name, i, reg_size));
            }
        }
    }

    // 然后处理所有语句中的操作数，提取实际使用的寄存器
    extract_registers_from_all_operands(stmt, registers);
    // extract_registers_from_all_operands(stmt, registers);
}
void RegisterAnalyzer::extract_registers_from_all_operands(
    const ptxemu::ir::StatementContext &stmt,
    std::unordered_set<RegisterInfo, RegisterInfoHash> &registers) {
    // 使用visit来访问不同类型的指令并提取操作数中的寄存器
    stmt.visit([&registers](const auto &instr) {
        using T = std::decay_t<decltype(instr)>;
        
        // 检查是否有operands成员
        if constexpr (requires { instr.operands; }) {
            for (const auto &op : instr.operands) {
                extract_register_from_operand(op, registers);
            }
        }
    });
}

void RegisterAnalyzer::extract_register_from_operand(
    const ptxemu::ir::OperandContext &op,
    std::unordered_set<RegisterInfo, RegisterInfoHash> &registers) {
    // 检查操作数是否为寄存器类型
    if (std::holds_alternative<RegOperand>(op.data)) {
        const auto &reg = std::get<RegOperand>(op.data);
        std::string full_name = reg.fullName();
        
        // 根据寄存器名称前缀确定大小
        size_t reg_size = 4; // 默认32位
        if (full_name.rfind("rd", 0) == 0 || full_name.rfind("dp", 0) == 0) {
            reg_size = 8; // 64位寄存器
        } else if (full_name.rfind("p", 0) == 0) {
            reg_size = 1; // predicate is 1 bit but usually stored as 8 bits
        }
        
        registers.insert(RegisterInfo(full_name, -1, reg_size));
    } else if (std::holds_alternative<VecOperand>(op.data)) {
        // 对于向量操作数，递归处理其中的每个元素
        const auto &vec = std::get<VecOperand>(op.data);
        for (const auto &elem : vec.elements) {
            extract_register_from_operand(elem, registers);
        }

}

// void RegisterAnalyzer::extract_register_from_operand(
    }


// void RegisterAnalyzer::extract_registers_from_operand(
//     const OperandContext &op, const std::vector<Qualifier> &qualifiers,
//     std::unordered_set<RegisterInfo, RegisterInfoHash> &registers) {
//     if (op.operandType == O_REG) {
//         auto *reg = static_cast<OperandContext::REG *>(op.operand);
//         if (reg) {
//             // 获取寄存器的大小，根据qualifier确定
//             size_t reg_size = getBytes(qualifiers);
//             if (reg_size == 0) {
//                 // 如果没有明确的大小，使用默认的32位大小
//                 reg_size = sizeof(uint32_t);
//             }
//             // 添加寄存器到集合中，使用完整名称
//             std::string full_name =
//                 reg->regName; // + std::to_string(reg->regIdx);
//             registers.insert(RegisterInfo(full_name, reg->regIdx, reg_size));
//         }
//     } else if (op.operandType == O_VEC) {
//         // 对于向量操作数，递归处理其中的每个元素
//         auto *vec = static_cast<OperandContext::VEC *>(op.operand);
//         if (vec) {
//             for (const auto &element : vec->vec) {
//                 extract_registers_from_operand(element, qualifiers,
//                 registers);
//             }
//         }
//     }
//     // 其他类型如O_VAR, O_IMM, O_FA, O_PRED不需要分配寄存器空间
// }