#ifndef INSTRUCTION_PROCESSOR_UTILS_H
#define INSTRUCTION_PROCESSOR_UTILS_H

#include "ptxsim/thread_context.h"
#include "ptx_ir/ptx_types.h"

// 定义处理单操作数指令的宏
#define PROCESS_OPERATION_1(context, dst, qualifiers, reg) \
    do { \
        process_operation(context, dst, qualifiers); \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info, "reg")) { \
            if (reg) context->trace_register(reg, dst, qualifiers, true); \
        } \
    } while (0)

// 定义处理双操作数指令的宏
#define PROCESS_OPERATION_2(context, dst, src, qualifiers, reg) \
    do { \
        process_operation(context, dst, src, qualifiers); \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info, "reg")) { \
            if (reg) context->trace_register(reg, dst, qualifiers, true); \
        } \
    } while (0)

// 定义处理三操作数指令的宏
#define PROCESS_OPERATION_3(context, dst, src1, src2, qualifiers, reg) \
    do { \
        process_operation(context, dst, src1, src2, qualifiers); \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info, "reg")) { \
            if (reg) context->trace_register(reg, dst, qualifiers, true); \
        } \
    } while (0)

// 定义处理四操作数指令的宏
#define PROCESS_OPERATION_4(context, dst, src1, src2, src3, qualifiers, reg) \
    do { \
        process_operation(context, dst, src1, src2, src3, qualifiers); \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info, "reg")) { \
            if (reg) context->trace_register(reg, dst, qualifiers, true); \
        } \
    } while (0)

#endif // INSTRUCTION_PROCESSOR_UTILS_H