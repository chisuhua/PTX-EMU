#ifndef PTX_DEBUGGER_H
#define PTX_DEBUGGER_H

#include "ptx_config.h"
#include "debug_format.h"
#include "../utils/logger.h"
#include <any>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// 添加PTX_INFO_THREAD宏定义，确保在PTXDebugger类中可以使用
#ifndef PTX_INFO_THREAD
#define PTX_INFO_THREAD(fmt, ...)                                              \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,    \
                                                   "thread")) {                \
            auto formatted_msg = ptxsim::detail::printf_format(fmt, ##__VA_ARGS__); \
            ptxsim::detail::output_log_simple(ptxsim::log_level::info, "thread", \
                                 formatted_msg);                               \
        }                                                                      \
    } while (0)
#endif

#ifndef PTX_TRACE_EXEC
#define PTX_TRACE_EXEC(...)                                                    \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,    \
                                                   "exec")) {                  \
            auto formatted_msg = ptxsim::detail::printf_format(__VA_ARGS__);   \
            ptxsim::detail::output_log_simple(ptxsim::log_level::info, "exec", \
                                 formatted_msg);                               \
        }                                                                      \
    } while (0)
#endif

// 添加PTX_TRACE_INSTR宏定义
#ifndef PTX_TRACE_INSTR
#define PTX_TRACE_INSTR(pc, opcode, operands, block, thread)                   \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::trace,   \
                                                   "instr")) {                 \
            auto formatted_msg = ptxsim::detail::printf_format(                 \
                "%s %s pc=%4d: %s %s",                                          \
                block.to_string().c_str(),                                      \
                thread.to_string().c_str(), pc,                                 \
                opcode.c_str(), operands.c_str());                              \
            ptxsim::detail::output_log_simple(ptxsim::log_level::trace, "instr", \
                                 formatted_msg);                               \
        }                                                                      \
    } while (0)
#endif

#ifndef PTX_TRACE_MEM
#define PTX_TRACE_MEM(...)                                                     \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,    \
                                                   "mem")) {                   \
            auto formatted_msg = ptxsim::detail::printf_format(__VA_ARGS__);  \
            ptxsim::detail::output_log_simple(ptxsim::log_level::info, "mem",  \
                                 formatted_msg);                               \
        }                                                                      \
    } while (0)
#endif

#ifndef PTX_TRACE_REG
#define PTX_TRACE_REG(...)                                                     \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,    \
                                                   "reg")) {                   \
            auto formatted_msg = ptxsim::detail::printf_format(__VA_ARGS__);    \
            ptxsim::detail::output_log_simple(ptxsim::log_level::info, "reg",  \
                                 formatted_msg);                               \
        }                                                                      \
    } while (0)
#endif

namespace ptxsim {

// 调试器类 - 用于执行调试相关操作
class PTXDebugger {
public:
    static PTXDebugger &get() {
        static PTXDebugger instance;
        return instance;
    }

    // 记录指令执行
    void trace_instruction(int pc, const std::string &opcode,
                           const std::string &operands, const Dim3 &block,
                           const Dim3 &thread) {
        auto &config = ptxsim::DebugConfig::get();

        // 检查PC范围
        if (!config.is_pc_traced(pc))
            return;

        // 检查指令类型
        ptxsim::InstructionType type = config.classify_instruction(opcode);
        if (!config.is_instruction_traced(type))
            return;

        // 构建完整指令
        std::string full_instruction = opcode;
        if (!operands.empty()) {
            full_instruction += " " + operands;
        }

        // 检查线程过滤器
        if (!config.should_trace_thread(block, thread))
            return;

        // 触发回调
        config.trigger_instruction_callback(pc, full_instruction);

        // 记录日志，包含线程和块索引信息
        PTX_TRACE_EXEC("%s %s pc=%4d: %s", block.to_string().c_str(),
                       thread.to_string().c_str(), pc,
                       full_instruction.c_str());
    }

    // 记录内存访问
    void trace_memory_access(bool is_write, const std::string &addr_expr,
                             uint64_t addr, size_t size, void *value = nullptr,
                             const Dim3 &block = Dim3(0, 0, 0),
                             const Dim3 &thread = Dim3(0, 0, 0)) {
        if (!ptxsim::DebugConfig::get().is_memory_traced())
            return;

        // 检查线程过滤器
        if (!ptxsim::DebugConfig::get().should_trace_thread(block, thread))
            return;

        const char *access_type = is_write ? "W" : "R";
        std::string addr_str = ptxsim::debug_format::format_address(addr);

        if (value) {
            // 尝试根据大小解释值
            std::string value_str;
            if (size == 4) {
                uint32_t u32 = *static_cast<uint32_t *>(value);
                float f32 = *static_cast<float *>(value);
                value_str = ptxsim::debug_format::format_u32(u32, true) +
                            " / " + ptxsim::debug_format::format_f32(f32);
            } else if (size == 8) {
                uint64_t u64 = *static_cast<uint64_t *>(value);
                double f64 = *static_cast<double *>(value);
                value_str = ptxsim::debug_format::format_u64(u64, true) +
                            " / " + ptxsim::debug_format::format_f64(f64);
            } else {
                value_str =
                    ptxsim::detail::printf_format("0x%p", (void *)value);
            }

            PTX_TRACE_MEM("%s %s        %s: [%s] = %s (size=%zu)",
                          block.to_string().c_str(), thread.to_string().c_str(),
                          access_type, addr_str.c_str(), value_str.c_str(),
                          size);
        } else {
            PTX_TRACE_MEM("%s %s         %s: [%s] (size=%zu)",
                          block.to_string().c_str(), thread.to_string().c_str(),
                          access_type, addr_str.c_str(), size);
        }
    }

    // 记录寄存器访问
    void trace_register_access(const std::string &reg_name,
                               const std::any &value, bool is_write,
                               const Dim3 &block = Dim3(0, 0, 0),
                               const Dim3 &thread = Dim3(0, 0, 0)) {
        if (!ptxsim::DebugConfig::get().is_register_traced() &&
            !ptxsim::DebugConfig::get().has_watchpoint(reg_name))
            return;

        // 检查线程过滤器
        if (!ptxsim::DebugConfig::get().should_trace_thread(block, thread))
            return;

        const char *access_type = is_write ? "W" : "R";
        std::string value_str =
            ptxsim::debug_format::format_register_value(value, true);

        PTX_TRACE_REG("%s %s        %s: %s = %s", block.to_string().c_str(),
                      thread.to_string().c_str(), access_type, reg_name.c_str(),
                      value_str.c_str());
    }

    // 线程状态转储
    template <typename T>
    void dump_thread_state(const std::string &prefix, const T &thread_ctx,
                           const Dim3 &block, const Dim3 &thread) {
        PTX_INFO_THREAD("%s - Block(%d,%d,%d) Thread(%d,%d,%d)", prefix.c_str(),
                        block.x, block.y, block.z, thread.x, thread.y,
                        thread.z);
        // 这里可以添加更详细的线程状态输出
    }

    // 检查断点
    // bool check_breakpoint(const ptx_instruction_address &pc,
    //                       const ptx_statement_context &context) {
    //     if (ptxsim::DebugConfig::get().has_breakpoint(pc, context)) {
    //         PTX_INFO_EMU("Hit breakpoint at PC=%d", pc);
    //         return true;
    //     }
    //     return false;
    // }

    // 性能统计
    class PerfStats {
    public:
        void record_instruction(const std::string &opcode) {
            std::lock_guard<std::mutex> lock(mutex_);
            total_instructions++;
            instruction_counts_[opcode]++;
        }

        void dump_stats(std::ostream &os) {
            std::lock_guard<std::mutex> lock(mutex_);

            os << "Performance Statistics:" << std::endl;
            os << "  Total instructions: " << total_instructions << std::endl;

            if (total_instructions == 0)
                return;

            os << "  Instruction distribution:" << std::endl;

            // 排序指令按执行次数
            std::vector<std::pair<std::string, uint64_t>> sorted_instructions(
                instruction_counts_.begin(), instruction_counts_.end());
            std::sort(sorted_instructions.begin(), sorted_instructions.end(),
                      [](const auto &a, const auto &b) {
                          return a.second > b.second;
                      });

            for (const auto &pair : sorted_instructions) {
                double percentage =
                    (static_cast<double>(pair.second) / total_instructions) *
                    100.0;
                os << "    " << pair.first << ": " << pair.second << " ("
                   << ptxsim::detail::printf_format("%.2f%%", percentage) << ")"
                   << std::endl;
            }
        }

        void reset() {
            std::lock_guard<std::mutex> lock(mutex_);
            total_instructions = 0;
            instruction_counts_.clear();
        }

        uint64_t get_total_instructions() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return total_instructions;
        }

        std::unordered_map<std::string, uint64_t>
        get_instruction_counts() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return instruction_counts_;
        }

    private:
        mutable std::mutex mutex_;
        uint64_t total_instructions = 0;
        std::unordered_map<std::string, uint64_t> instruction_counts_;
    };

    PerfStats &get_perf_stats() { return perf_stats_; }

    // 加载配置文件
    bool load_config(const std::string &filename) {
        bool success = true;

        if (ptxsim::LoggerConfig::get().load_from_file(filename)) {
            PTX_INFO_EMU("Loaded logger configuration from %s",
                         filename.c_str());
        } else {
            PTX_WARN_EMU("Failed to load logger configuration from %s",
                         filename.c_str());
            success = false;
        }

        if (ptxsim::DebugConfig::get().load_from_file(filename)) {
            PTX_INFO_EMU("Loaded debugger configuration from %s",
                         filename.c_str());
        } else {
            PTX_WARN_EMU("Failed to load debugger configuration from %s",
                         filename.c_str());
            success = false;
        }

        return success;
    }

private:
    PTXDebugger() = default;
    PerfStats perf_stats_;

    PTXDebugger(const PTXDebugger &) = delete;
    PTXDebugger &operator=(const PTXDebugger &) = delete;
};

} // namespace ptxsim

#define PTX_DUMP_THREAD_STATE(prefix, thread_ctx, block, thread)               \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,    \
                                                   "thread")) {                \
            ptxsim::PTXDebugger::get().dump_thread_state(prefix, thread_ctx,   \
                                                         block, thread);       \
        }                                                                      \
    } while (0)

#endif // PTX_DEBUGGER_H