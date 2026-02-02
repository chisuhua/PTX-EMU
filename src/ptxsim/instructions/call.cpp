// Helper function to handle printf calls
#include "memory/hardware_memory_manager.h" // 添加HardwareMemoryManager头文件
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include <iostream>

void RET::process_operation(ThreadContext *context) {
    // 如果调用栈为空，表示这是最外层的返回，设置线程状态为退出
    if (context->call_stack.empty()) {
        context->state = EXIT;
    } else {
        // 从调用栈弹出上一个PC地址，回到调用点
        int return_pc = context->call_stack.top();
        context->call_stack.pop();
        context->next_pc = return_pc;
    }
}

void CALL::process_operation(ThreadContext *context, void *op[3],
                             const std::vector<Qualifier> &qualifier) {
    // 第一个操作数是函数标签/地址，第二个操作数可能是参数信息（这里简化处理）
    // 第三个操作数是返回地址，即当前指令的下一条指令地址
    int target_pc = *(int *)(op[0]); // 函数入口地址
    int return_pc = context->pc + 1; // 当前指令的下一条指令作为返回地址

    // 将返回地址压入调用栈
    context->call_stack.push(return_pc);

    // 跳转到函数入口
    context->next_pc = target_pc;
}

// 辅助函数：从内存中获取字符串
std::string get_string_from_memory(ThreadContext *context, uint64_t address) {
    // 这是一个简化实现，实际上我们需要从适当的内存空间获取字符串
    // 由于当前架构限制，这里只是一个占位符实现
    // 在实际实现中，这需要通过内存管理器从全局内存中读取字符串
    static std::string temp_str = "Placeholder string";
    return temp_str;
}

void CALL::handlePrintf(ThreadContext *context, StatementContext &stmt) {
    auto ss = std::get<CallInstr>(stmt.data);

    if (ss.operands.size() < 1) {
        return; // Need at least format string
    }

    // 获取格式字符串
    std::string formatStr;
    void *formatAddr = context->acquire_operand(ss.operands[0], ss.qualifiers);
    if (!formatAddr) {
        return; // 无法获取格式字符串
    }

    // 假设格式字符串是64位地址
    uint64_t formatPtr = *static_cast<uint64_t *>(formatAddr);

    // 从内存中获取格式字符串 - 直接通过warp_context访问内存
    formatStr = get_string_from_memory(context, formatPtr);

    // 准备参数列表
    std::vector<void *> args;
    for (int i = 1; i < ss.operands.size(); i++) {
        void *argAddr = context->acquire_operand(ss.operands[i], ss.qualifiers);
        if (argAddr) {
            args.push_back(argAddr);
        }
    }

    // 执行格式化输出
    parseAndPrintFormat(context, formatStr, args);
}

void CALL::parseAndPrintFormat(ThreadContext *context,
                               const std::string &format,
                               const std::vector<void *> &args) {
    std::string result;
    size_t argIndex = 0;

    for (size_t i = 0; i < format.length(); i++) {
        if (format[i] == '%' && i + 1 < format.length()) {
            // 处理格式说明符
            i++; // 移过%

            // 跳过标志、宽度、精度（为了简化）
            while (format[i] &&
                   (format[i] == '-' || format[i] == '+' || format[i] == '#' ||
                    format[i] == ' ' || format[i] == '*' ||
                    (format[i] >= '0' && format[i] <= '9'))) {
                i++;
            }

            if (format[i] == '%') {
                // 字面量 %
                result += '%';
                continue;
            }

            if (argIndex >= args.size()) {
                break; // 参数不够
            }

            // 根据格式说明符确定参数类型
            switch (format[i]) {
            case 'd':
            case 'i': // 有符号整数
            {
                int val = *static_cast<int *>(args[argIndex]);
                result += std::to_string(val);
            } break;
            case 'u': // 无符号整数
            {
                unsigned int val = *static_cast<unsigned int *>(args[argIndex]);
                result += std::to_string(val);
            } break;
            case 'x':
            case 'X': // 十六进制
            {
                unsigned int val = *static_cast<unsigned int *>(args[argIndex]);
                char buf[32];
                snprintf(buf, sizeof(buf), (format[i] == 'x') ? "%x" : "%X",
                         val);
                result += buf;
            } break;
            case 'o': // 八进制
            {
                unsigned int val = *static_cast<unsigned int *>(args[argIndex]);
                char buf[32];
                snprintf(buf, sizeof(buf), "%o", val);
                result += buf;
            } break;
            case 'f':
            case 'F': // 浮点数
            {
                float val = *static_cast<float *>(args[argIndex]);
                char buf[64];
                snprintf(buf, sizeof(buf), "%.6f", val);
                result += buf;
            } break;
            case 'e':
            case 'E': // 科学计数法
            {
                float val = *static_cast<float *>(args[argIndex]);
                char buf[64];
                snprintf(buf, sizeof(buf), "%e", val);
                result += buf;
            } break;
            case 'g':
            case 'G': // 一般格式
            {
                float val = *static_cast<float *>(args[argIndex]);
                char buf[64];
                snprintf(buf, sizeof(buf), "%g", val);
                result += buf;
            } break;
            case 'c': // 字符
            {
                int val = *static_cast<int *>(args[argIndex]);
                result += static_cast<char>(val);
            } break;
            case 's': // 字符串
            {
                // 在PTX中，字符串是指向内存的指针
                uint64_t str_ptr = *static_cast<uint64_t *>(args[argIndex]);
                std::string str = get_string_from_memory(context, str_ptr);
                result += str;
            } break;
            case 'p': // 指针
            {
                uint64_t ptr = *static_cast<uint64_t *>(args[argIndex]);
                char buf[32];
                snprintf(buf, sizeof(buf), "0x%lx", ptr);
                result += buf;
            } break;
            default:
                // 未知格式，直接输出
                result += '%';
                result += format[i];
                continue;
            }

            argIndex++;
        } else {
            result += format[i];
        }
    }

    printf("%s", result.c_str());
    fflush(stdout);
}