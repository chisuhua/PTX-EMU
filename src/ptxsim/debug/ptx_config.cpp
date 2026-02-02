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
    return statement.toString();

    // std::ostringstream oss;

    //     // 根据statement的类型进行不同的处理
    //     switch (statement.type) {
    //         // 处理通用指令类型，这些指令继承自BASE_INSTR，具有operands字段
    // #define X(enum_val, type_name, str, op_count, struct_kind) \
//     case enum_val: { \
//         if constexpr (std::string_view(#struct_kind) == "GENERIC_INSTR"
    //         ||     \
//                       std::string_view(#struct_kind) == "ATOM_INSTR" || \
//                       std::string_view(#struct_kind) ==
    //                       "PREDICATE_PREFIX" ||  \
//                       std::string_view(#struct_kind) == "WMMA_INSTR") { \
//             auto *stmt = static_cast<const StatementContext::BASE_INSTR
    //             *>(    \
//                 statement.statement); \
//             if (stmt && !stmt->operands.empty()) { \
//                 for (size_t i = 0; i < stmt->operands.size(); ++i) { \
//                     if (i > 0) \
//                         oss << ", "; \
//                     oss << stmt->operands[i].toString(); \
//                 } \
//             } \
//         } else { \
//             /* 其他类型指令的处理 */                                  \
//         } \
//         break; \
//     }
    // #include "ptx_ir/ptx_op.def"
    // #undef X
    //     default:
    //         break;
    //     }

    // return oss.str();
}

std::string
DebugConfig::get_full_instruction_string(const StatementContext &statement) {
    // 使用StatementContext的相关方法来获取指令字符串
    std::ostringstream oss;
    oss << S2s(statement.type) << " ";

    // 根据不同的type获取对应的操作数
    std::string operands = getOperandsString(statement);
    oss << operands;

    return oss.str();
}

// 从INI配置中加载调试器配置
void DebugConfig::load_from_ini_section(
    const inipp::Ini<char>::Section &debugger_section) {
    // 按类型设置指令跟踪 - 使用辅助函数减少重复
    // set_instruction_type_trace(debugger_section,
    //                            "trace_instruction_type.memory",
    //                            InstructionType::MEMORY);
    // set_instruction_type_trace(debugger_section,
    //                            "trace_instruction_type.arithmetic",
    //                            InstructionType::ARITHMETIC);
    // set_instruction_type_trace(debugger_section,
    //                            "trace_instruction_type.control",
    //                            InstructionType::CONTROL);
    // set_instruction_type_trace(debugger_section,
    // "trace_instruction_type.logic",
    //                            InstructionType::LOGIC);
    // set_instruction_type_trace(debugger_section,
    //                            "trace_instruction_type.convert",
    //                            InstructionType::CONVERT);
    // set_instruction_type_trace(debugger_section,
    //                            "trace_instruction_type.special",
    //                            InstructionType::SPECIAL);
    // set_instruction_type_trace(debugger_section,
    // "trace_instruction_type.other",
    //                            InstructionType::OTHER);

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

    // 设置warp跟踪
    std::string trace_warp_str;
    inipp::get_value(debugger_section, "trace_warp", trace_warp_str);
    if (!trace_warp_str.empty()) {
        bool trace_warp = (trace_warp_str == "true" || trace_warp_str == "1");
        set_trace_warp_enabled(trace_warp);
    }

    // 设置指令状态跟踪
    std::string trace_instr_status_str;
    inipp::get_value(debugger_section, "trace_instruction_status",
                     trace_instr_status_str);
    if (!trace_instr_status_str.empty()) {
        bool trace_instr_status =
            (trace_instr_status_str == "true" || trace_instr_status_str == "1");
        set_trace_instruction_status_enabled(trace_instr_status);
    }

    // 设置lane跟踪掩码
    std::string trace_lanes_str;
    inipp::get_value(debugger_section, "trace_lanes", trace_lanes_str);
    if (!trace_lanes_str.empty()) {
        set_trace_lanes_mask_from_string(trace_lanes_str);
    }
}

} // namespace ptxsim