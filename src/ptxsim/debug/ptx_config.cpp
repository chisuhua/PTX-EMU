#include "ptxsim/ptx_config.h"
#include "inipp/inipp.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "utils/logger.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>

namespace ptxsim {

// 辅助函数：获取操作数字符串
std::string DebugConfig::getOperandsString(const StatementContext &statement) {
    std::ostringstream oss;

    // 根据statement的类型进行不同的处理
    switch (statement.statementType) {
        // 处理通用指令类型，这些指令继承自BASE_INSTR，具有operands字段
#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    case enum_val: {                                                           \
        if constexpr (std::string_view(#struct_kind) == "GENERIC_INSTR" ||     \
                      std::string_view(#struct_kind) == "ATOM_INSTR" ||        \
                      std::string_view(#struct_kind) == "WMMA_INSTR") {        \
            auto *stmt = static_cast<const StatementContext::BASE_INSTR *>(    \
                statement.statement);                                          \
            if (stmt && !stmt->operands.empty()) {                             \
                for (size_t i = 0; i < stmt->operands.size(); ++i) {           \
                    if (i > 0)                                                 \
                        oss << ", ";                                           \
                    oss << stmt->operands[i].toString();                       \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            /* 其他类型指令的处理 */                                  \
        }                                                                      \
        break;                                                                 \
    }
#include "ptx_ir/ptx_op.def"
#undef X
    default:
        break;
    }

    return oss.str();
}

std::string
DebugConfig::get_full_instruction_string(const StatementContext &statement) {
    // 使用StatementContext的相关方法来获取指令字符串
    std::ostringstream oss;
    oss << S2s(statement.statementType) << " ";

    // 根据不同的statementType获取对应的操作数
    std::string operands = getOperandsString(statement);
    oss << operands;

    return oss.str();
}

// 处理GENERIC_INSTR类型的指令操作数（此函数是通用处理，上面的constexpr已经处理了）
// void DebugConfig::handle_GENERIC_INSTR_operands(
//     std::ostringstream &oss, const StatementContext &statement) {
//     // 此函数实际上不会被调用，因为constexpr条件在运行时就已经确定了
//     // 但为了完整性保留此函数
//     auto *stmt =
//         static_cast<const StatementContext::BASE_INSTR
//         *>(statement.statement);
//     if (stmt && !stmt->operands.empty()) {
//         for (size_t i = 0; i < stmt->operands.size(); ++i) {
//             if (i > 0)
//                 oss << ", ";
//             oss << stmt->operands[i].to_string();
//         }
//     }
// }

// // 处理ATOM_INSTR类型的指令操作数
// void DebugConfig::handle_ATOM_INSTR_operands(
//     std::ostringstream &oss, const StatementContext &statement) {
//     auto *stmt =
//         static_cast<const StatementContext::ATOM *>(statement.statement);
//     if (stmt && !stmt->operands.empty()) {
//         for (size_t i = 0; i < stmt->operands.size(); ++i) {
//             if (i > 0)
//                 oss << ", ";
//             oss << stmt->operands[i].to_string();
//         }
//     }
// }

// // 处理WMMA_INSTR类型的指令操作数
// void DebugConfig::handle_WMMA_INSTR_operands(
//     std::ostringstream &oss, const StatementContext &statement) {
//     auto *stmt =
//         static_cast<const StatementContext::WMMA *>(statement.statement);
//     if (stmt && !stmt->operands.empty()) {
//         for (size_t i = 0; i < stmt->operands.size(); ++i) {
//             if (i > 0)
//                 oss << ", ";
//             oss << stmt->operands[i].to_string();
//         }
//     }
// }

// // 处理BRANCH类型的指令操作数
// void DebugConfig::handle_BRANCH_operands(std::ostringstream &oss,
//                                          const StatementContext &statement) {
//     auto *stmt =
//         static_cast<const StatementContext::BRA *>(statement.statement);
//     if (stmt) {
//         oss << stmt->braTarget;
//     }
// }

// // 处理SIMPLE_NAME类型的指令操作数
// void DebugConfig::handle_SIMPLE_NAME_operands(
//     std::ostringstream &oss, const StatementContext &statement) {
//     auto *stmt =
//         static_cast<const StatementContext::DOLLOR *>(statement.statement);
//     if (stmt) {
//         oss << "$" << stmt->dollorName;
//     }
// }

// // 处理SIMPLE_STRING类型的指令操作数
// void DebugConfig::handle_SIMPLE_STRING_operands(
//     std::ostringstream &oss, const StatementContext &statement) {
//     auto *stmt =
//         static_cast<const StatementContext::PRAGMA *>(statement.statement);
//     if (stmt) {
//         oss << "\"" << stmt->pragmaString << "\"";
//     }
// }

// // 处理PREDICATE_PREFIX类型的指令操作数
// void DebugConfig::handle_PREDICATE_PREFIX_operands(
//     std::ostringstream &oss, const StatementContext &statement) {
//     auto *stmt = static_cast<const StatementContext::AT
//     *>(statement.statement); if (stmt) {
//         oss << stmt->atPred.to_string() << ", " << stmt->atLabelName;
//     }
// }

// // 处理BARRIER类型的指令操作数
// void DebugConfig::handle_BARRIER_operands(std::ostringstream &oss,
//                                           const StatementContext &statement)
//                                           {
//     auto *stmt =
//         static_cast<const StatementContext::BAR *>(statement.statement);
//     if (stmt) {
//         oss << stmt->barType << ", " << stmt->barId;
//     }
// }

// // 处理VOID_INSTR类型的指令操作数（无操作数）
// void DebugConfig::handle_VOID_INSTR_operands(
//     std::ostringstream &oss, const StatementContext &statement) {
//     // 无操作数
// }

// // 处理OPERAND_REG类型的指令操作数
// void DebugConfig::handle_OPERAND_REG_operands(
//     std::ostringstream &oss, const StatementContext &statement) {
//     auto *stmt =
//         static_cast<const StatementContext::REG *>(statement.statement);
//     if (stmt) {
//         oss << "%" << stmt->regName << "[" << stmt->regNum << "]";
//     }
// }

// // 处理OPERAND_CONST类型的指令操作数
// void DebugConfig::handle_OPERAND_CONST_operands(
//     std::ostringstream &oss, const StatementContext &statement) {
//     auto *stmt =
//         static_cast<const StatementContext::CONST *>(statement.statement);
//     if (stmt) {
//         oss << "[" << stmt->constName << "]";
//     }
// }

// // 处理OPERAND_MEMORY类型的指令操作数
// void DebugConfig::handle_OPERAND_MEMORY_operands(
//     std::ostringstream &oss, const StatementContext &statement) {
//     // 分别处理SHARED和LOCAL
//     auto *shared_stmt =
//         static_cast<const StatementContext::SHARED *>(statement.statement);
//     if (shared_stmt) {
//         oss << ".shared [" << shared_stmt->name << "]";
//     } else {
//         auto *local_stmt =
//             static_cast<const StatementContext::LOCAL
//             *>(statement.statement);
//         if (local_stmt) {
//             oss << ".local [" << local_stmt->name << "]";
//         }
//     }
// }

// 从INI配置中加载调试器配置
void DebugConfig::load_from_ini_section(
    const inipp::Ini<char>::Section &debugger_section) {
    // 按类型设置指令跟踪 - 使用辅助函数减少重复
    set_instruction_type_trace(debugger_section,
                               "trace_instruction_type.memory",
                               InstructionType::MEMORY);
    set_instruction_type_trace(debugger_section,
                               "trace_instruction_type.arithmetic",
                               InstructionType::ARITHMETIC);
    set_instruction_type_trace(debugger_section,
                               "trace_instruction_type.control",
                               InstructionType::CONTROL);
    set_instruction_type_trace(debugger_section, "trace_instruction_type.logic",
                               InstructionType::LOGIC);
    set_instruction_type_trace(debugger_section,
                               "trace_instruction_type.convert",
                               InstructionType::CONVERT);
    set_instruction_type_trace(debugger_section,
                               "trace_instruction_type.special",
                               InstructionType::SPECIAL);
    set_instruction_type_trace(debugger_section, "trace_instruction_type.other",
                               InstructionType::OTHER);

    // 设置内存和寄存器跟踪
    std::string trace_mem;
    inipp::get_value(debugger_section, "trace_memory", trace_mem);
    if (!trace_mem.empty()) {
        enable_memory_trace((trace_mem == "true" || trace_mem == "1"));
    }

    std::string trace_reg;
    inipp::get_value(debugger_section, "trace_registers", trace_reg);
    if (!trace_reg.empty()) {
        enable_register_trace((trace_reg == "true" || trace_reg == "1"));
    }
}

// 从配置文件加载
bool DebugConfig::load_from_file(const std::string &filename) {
    std::lock_guard<std::mutex> lock(get_mutex());
    try {
        inipp::Ini<char> ini;
        std::ifstream is(filename);
        if (!is.is_open()) {
            PTX_WARN_EMU("Could not open configuration file: %s",
                         filename.c_str());
            return false;
        }

        ini.parse(is);

        // 读取debugger section
        auto debugger_section = ini.sections["debugger"];

        std::string trace_instr;
        inipp::get_value(debugger_section, "trace_instruction", trace_instr);
        if (!trace_instr.empty()) {
            bool trace = (trace_instr == "true" || trace_instr == "1");
            // 批量设置所有指令类型的跟踪
            for (int i = 0; i <= static_cast<int>(InstructionType::OTHER);
                 ++i) {
                enable_instruction_trace(static_cast<InstructionType>(i),
                                         trace);
            }
        }

        // 加载调试器特定配置
        load_from_ini_section(debugger_section);

        PTX_INFO_EMU("Debugger configuration loaded from %s", filename.c_str());
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error loading debugger config: " << e.what() << std::endl;
        return false;
    }
}

} // namespace ptxsim