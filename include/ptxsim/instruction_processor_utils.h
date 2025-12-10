#ifndef INSTRUCTION_PROCESSOR_UTILS_H
#define INSTRUCTION_PROCESSOR_UTILS_H

#include "ptxsim/ptx_debug.h"

// 定义宏，用于包装process_operation调用并在满足条件时更新寄存器跟踪
// 适用于1个目标和1个源操作数的指令 (dst, src)
#define PROCESS_OPERATION_2(context, dst, src, qualifiers, reg_operand)        \
    do {                                                                       \
        process_operation(context, dst, src, qualifiers);                      \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,    \
                                                   "reg")) {                   \
            context->update_register(reg_operand, dst, qualifiers);            \
        }                                                                      \
    } while (0)

// 定义宏，用于包装process_operation调用并在满足条件时更新寄存器跟踪
// 适用于1个目标和2个源操作数的指令 (dst, src1, src2)
#define PROCESS_OPERATION_3(context, dst, src1, src2, qualifiers, reg_operand) \
    do {                                                                       \
        process_operation(context, dst, src1, src2, qualifiers);               \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,    \
                                                   "reg")) {                   \
            context->update_register(reg_operand, dst, qualifiers);            \
        }                                                                      \
    } while (0)

// 定义宏，用于包装process_operation调用并在满足条件时更新寄存器跟踪
// 适用于1个目标和3个源操作数的指令 (dst, src1, src2, src3)
#define PROCESS_OPERATION_4(context, dst, src1, src2, src3, qualifiers,        \
                            reg_operand)                                       \
    do {                                                                       \
        process_operation(context, dst, src1, src2, src3, qualifiers);         \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,    \
                                                   "reg")) {                   \
            context->update_register(reg_operand, dst, qualifiers);            \
        }                                                                      \
    } while (0)

#endif // INSTRUCTION_PROCESSOR_UTILS_H