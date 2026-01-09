#ifndef PTX_DEBUG_H
#define PTX_DEBUG_H

#include "inipp/inipp.h"
#include "ptx_ir/statement_context.h"
#include "utils/logger.h"
#include <any>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ptxsim {

// 指令类型枚举
enum class InstructionType {
    MEMORY,     // 内存操作 (LD, ST)
    ARITHMETIC, // 算术操作 (ADD, MUL, etc.)
    CONTROL,    // 控制流 (BRA, RET, etc.)
    LOGIC,      // 逻辑操作 (AND, OR, etc.)
    CONVERT,    // 类型转换
    SPECIAL,    // 特殊指令
    OTHER
};

// 断点条件类型
using BreakpointCondition = std::function<bool(
    int pc, const std::unordered_map<std::string, std::any> &context)>;

// 用于表示要trace的block和thread索引
struct ThreadFilter {
    int block_x = -1; // -1 表示所有值
    int block_y = -1;
    int block_z = -1;
    int thread_x = -1;
    int thread_y = -1;
    int thread_z = -1;

    bool matches(const Dim3 &block_idx, const Dim3 &thread_idx) const {
        return (block_x == -1 || block_idx.x == block_x) &&
               (block_y == -1 || block_idx.y == block_y) &&
               (block_z == -1 || block_idx.z == block_z) &&
               (thread_x == -1 || thread_idx.x == thread_x) &&
               (thread_y == -1 || thread_idx.y == thread_y) &&
               (thread_z == -1 || thread_idx.z == thread_z);
    }
};

// 调试配置
class DebugConfig {
public:
    static DebugConfig &get() {
        static DebugConfig instance;
        return instance;
    }

    // 添加获取完整指令字符串的函数声明
    static std::string
    get_full_instruction_string(const StatementContext &statement);

    // 启用/禁用特定类型的指令跟踪
    void enable_instruction_trace(InstructionType type, bool enable = true) {
        std::lock_guard<std::mutex> lock(get_mutex());
        instruction_tracing_[type] = enable;
    }

    // 检查是否启用特定类型的指令跟踪
    bool is_instruction_traced(InstructionType type) const {
        std::lock_guard<std::mutex> lock(
            const_cast<DebugConfig *>(this)->get_mutex());
        auto it = instruction_tracing_.find(type);
        return it != instruction_tracing_.end() && it->second;
    }

    // 启用/禁用内存访问跟踪
    void enable_memory_trace(bool enable = true) {
        std::lock_guard<std::mutex> lock(get_mutex());
        trace_memory_ = enable;
    }

    bool is_memory_traced() const {
        std::lock_guard<std::mutex> lock(
            const_cast<DebugConfig *>(this)->get_mutex());
        return trace_memory_;
    }

    // 启用/禁用寄存器访问跟踪
    void enable_register_trace(bool enable = true) {
        std::lock_guard<std::mutex> lock(get_mutex());
        trace_registers_ = enable;
    }

    bool is_register_traced() const {
        std::lock_guard<std::mutex> lock(
            const_cast<DebugConfig *>(this)->get_mutex());
        return trace_registers_;
    }

    // 设置线程过滤器 - 仅对指定的blockIdx和threadIdx进行trace
    void set_thread_filter(int block_x = -1, int block_y = -1, int block_z = -1,
                           int thread_x = -1, int thread_y = -1,
                           int thread_z = -1) {
        std::lock_guard<std::mutex> lock(get_mutex());
        thread_filter_.block_x = block_x;
        thread_filter_.block_y = block_y;
        thread_filter_.block_z = block_z;
        thread_filter_.thread_x = thread_x;
        thread_filter_.thread_y = thread_y;
        thread_filter_.thread_z = thread_z;
        has_thread_filter_ = true;
    }

    // 检查是否应该跟踪指定线程
    bool should_trace_thread(const Dim3 &block_idx,
                             const Dim3 &thread_idx) const {
        std::lock_guard<std::mutex> lock(
            const_cast<DebugConfig *>(this)->get_mutex());
        if (!has_thread_filter_) {
            return true; // 如果没有设置过滤器，则跟踪所有线程
        }
        return thread_filter_.matches(block_idx, thread_idx);
    }

    // 设置断点
    void set_breakpoint(int pc) {
        std::lock_guard<std::mutex> lock(get_mutex());
        breakpoints_[pc] = nullptr; // 无条件断点
    }

    // 设置条件断点
    void set_conditional_breakpoint(int pc, BreakpointCondition condition) {
        std::lock_guard<std::mutex> lock(get_mutex());
        breakpoints_[pc] = condition;
    }

    void clear_breakpoint(int pc) {
        std::lock_guard<std::mutex> lock(get_mutex());
        breakpoints_.erase(pc);
    }

    // 检查是否有断点
    bool has_breakpoint(
        int pc,
        const std::unordered_map<std::string, std::any> &context = {}) const {
        std::lock_guard<std::mutex> lock(
            const_cast<DebugConfig *>(this)->get_mutex());
        auto it = breakpoints_.find(pc);
        if (it != breakpoints_.end()) {
            if (it->second) { // 有条件断点
                return it->second(pc, context);
            } else { // 无条件断点
                return true;
            }
        }
        return false;
    }

    // 设置PC范围跟踪
    void set_pc_range(int start, int end) {
        std::lock_guard<std::mutex> lock(get_mutex());
        pc_start_ = start;
        pc_end_ = end;
    }

    // 检查PC是否在跟踪范围内
    bool is_pc_traced(int pc) const {
        std::lock_guard<std::mutex> lock(
            const_cast<DebugConfig *>(this)->get_mutex());
        return pc >= pc_start_ && pc <= pc_end_;
    }

    // 设置指令回调
    void set_instruction_callback(
        std::function<void(int, const std::string &)> callback) {
        std::lock_guard<std::mutex> lock(get_mutex());
        instruction_callback_ = callback;
    }

    // 触发指令回调
    void trigger_instruction_callback(int pc, const std::string &instruction) {
        std::lock_guard<std::mutex> lock(get_mutex());
        if (instruction_callback_) {
            instruction_callback_(pc, instruction);
        }
    }

    // 从配置文件加载
    bool load_from_file(const std::string &filename) {
        std::lock_guard<std::mutex> lock(get_mutex());
        try {
            inipp::Ini<char> ini;
            std::ifstream is(filename);
            if (!is.is_open()) {
                return false;
            }

            ini.parse(is);

            // 读取debugger section
            auto debugger_section = ini.sections["debugger"];

            std::string trace_instr;
            inipp::get_value(debugger_section, "trace_instruction",
                             trace_instr);
            if (!trace_instr.empty()) {
                bool trace = (trace_instr == "true" || trace_instr == "1");
                // 批量设置所有指令类型的跟踪
                for (int i = 0; i <= static_cast<int>(InstructionType::OTHER); ++i) {
                    enable_instruction_trace(static_cast<InstructionType>(i), trace);
                }
            }

            // 按类型设置指令跟踪 - 使用辅助函数减少重复
            set_instruction_type_trace(debugger_section, "trace_instruction_type.memory", InstructionType::MEMORY);
            set_instruction_type_trace(debugger_section, "trace_instruction_type.arithmetic", InstructionType::ARITHMETIC);
            set_instruction_type_trace(debugger_section, "trace_instruction_type.control", InstructionType::CONTROL);
            set_instruction_type_trace(debugger_section, "trace_instruction_type.logic", InstructionType::LOGIC);
            set_instruction_type_trace(debugger_section, "trace_instruction_type.convert", InstructionType::CONVERT);
            set_instruction_type_trace(debugger_section, "trace_instruction_type.special", InstructionType::SPECIAL);
            set_instruction_type_trace(debugger_section, "trace_instruction_type.other", InstructionType::OTHER);

            // 设置内存和寄存器跟踪
            std::string trace_mem;
            inipp::get_value(debugger_section, "trace_memory", trace_mem);
            if (!trace_mem.empty()) {
                enable_memory_trace((trace_mem == "true" || trace_mem == "1"));
            }

            std::string trace_reg;
            inipp::get_value(debugger_section, "trace_registers", trace_reg);
            if (!trace_reg.empty()) {
                enable_register_trace(
                    (trace_reg == "true" || trace_reg == "1"));
            }

            return true;
        } catch (const std::exception &e) {
            std::cerr << "Error loading debugger config: " << e.what()
                      << std::endl;
            return false;
        }
    }

    // 检查是否有watchpoint（观察点）
    bool has_watchpoint(const std::string &reg_name) const {
        std::lock_guard<std::mutex> lock(
            const_cast<DebugConfig *>(this)->get_mutex());
        return watchpoints_.find(reg_name) != watchpoints_.end();
    }

    // 添加观察点
    void add_watchpoint(const std::string &reg_name) {
        std::lock_guard<std::mutex> lock(get_mutex());
        watchpoints_.insert(reg_name);
    }

    // 移除观察点
    void remove_watchpoint(const std::string &reg_name) {
        std::lock_guard<std::mutex> lock(get_mutex());
        watchpoints_.erase(reg_name);
    }

    // 清除所有观察点
    void clear_watchpoints() {
        std::lock_guard<std::mutex> lock(get_mutex());
        watchpoints_.clear();
    }

    std::mutex &get_mutex() { return mutex_; }
    const std::mutex &get_mutex() const { return mutex_; }

private:
    // 辅助函数：设置特定类型的指令跟踪
    void set_instruction_type_trace(const inipp::Ini<char>::Section &section,
                                   const std::string &key, InstructionType type) {
        std::string value;
        inipp::get_value(section, key.c_str(), value);
        if (!value.empty()) {
            enable_instruction_trace(type, (value == "true" || value == "1"));
        }
    }

    mutable std::mutex mutex_; // 使用mutable确保const成员函数中也可访问
    std::unordered_map<InstructionType, bool> instruction_tracing_;
    bool trace_memory_ = false;
    bool trace_registers_ = false;
    std::unordered_map<int, BreakpointCondition> breakpoints_;
    int pc_start_ = 0;
    int pc_end_ = INT32_MAX;
    std::function<void(int, const std::string &)> instruction_callback_;

    // 线程过滤器
    ThreadFilter thread_filter_;
    bool has_thread_filter_ = false;

    // 观察点
    std::unordered_set<std::string> watchpoints_;

    DebugConfig() = default; // 添加默认构造函数
    DebugConfig(const DebugConfig &) = delete;
    DebugConfig &operator=(const DebugConfig &) = delete;

public:
    // 指令分类工具
    inline InstructionType classify_instruction(const std::string &opcode) {
        static const std::unordered_map<std::string, InstructionType>
            instruction_map = {
                // 内存操作
                {"ld", InstructionType::MEMORY},
                {"st", InstructionType::MEMORY},
                {"ldu", InstructionType::MEMORY},
                {"cvt", InstructionType::MEMORY}, // 某些cvt指令处理地址转换

                // 算术操作
                {"add", InstructionType::ARITHMETIC},
                {"sub", InstructionType::ARITHMETIC},
                {"mul", InstructionType::ARITHMETIC},
                {"div", InstructionType::ARITHMETIC},
                {"fma", InstructionType::ARITHMETIC},
                {"rem", InstructionType::ARITHMETIC},
                {"abs", InstructionType::ARITHMETIC},
                {"neg", InstructionType::ARITHMETIC},
                {"sqrt", InstructionType::ARITHMETIC},
                {"rcp", InstructionType::ARITHMETIC},
                {"rsqrt", InstructionType::ARITHMETIC},
                {"mad", InstructionType::ARITHMETIC},
                {"mad24", InstructionType::ARITHMETIC},
                {"mad.lo", InstructionType::ARITHMETIC},
                {"mad.hi", InstructionType::ARITHMETIC},

                // 控制流
                {"bra", InstructionType::CONTROL},
                {"call", InstructionType::CONTROL},
                {"ret", InstructionType::CONTROL},
                {"exit", InstructionType::CONTROL},
                {"bar", InstructionType::CONTROL},
                {"bra.uni", InstructionType::CONTROL},
                {"break", InstructionType::CONTROL},
                {"continue", InstructionType::CONTROL},

                // 逻辑操作
                {"and", InstructionType::LOGIC},
                {"or", InstructionType::LOGIC},
                {"xor", InstructionType::LOGIC},
                {"not", InstructionType::LOGIC},
                {"shl", InstructionType::LOGIC},
                {"shr", InstructionType::LOGIC},
                {"setp", InstructionType::LOGIC},
                {"selp", InstructionType::LOGIC},
                {"slct", InstructionType::LOGIC},

                // 类型转换
                {"cvt", InstructionType::CONVERT},
                {"mov", InstructionType::CONVERT},

                // 特殊指令
                {"tex", InstructionType::SPECIAL},
                {"atom", InstructionType::SPECIAL},
                {"vote", InstructionType::SPECIAL},
                {"trap", InstructionType::SPECIAL},
                {"brkpt", InstructionType::SPECIAL}};

        auto it = instruction_map.find(opcode);
        if (it != instruction_map.end()) {
            return it->second;
        }

        // 尝试匹配前缀
        for (const auto &pair : instruction_map) {
            if (opcode.find(pair.first) == 0) {
                return pair.second;
            }
        }

        return InstructionType::OTHER;
    }
}; // 结束DebugConfig类定义

// 格式化工具
namespace debug_format {
// 格式化32位整数（多种进制）
inline std::string format_i32(int32_t value, bool hex = false) {
    if (hex) {
        std::stringstream ss;
        ss << "0x" << std::hex << std::setw(8) << std::setfill('0') << value;
        return ss.str();
    }
    return detail::printf_format("%d", value);
}

// 格式化32位无符号整数
inline std::string format_u32(uint32_t value, bool hex = false) {
    if (hex) {
        std::stringstream ss;
        ss << "0x" << std::hex << std::setw(8) << std::setfill('0') << value;
        return ss.str();
    }
    return detail::printf_format("%u", value);
}

// 格式化64位整数
inline std::string format_i64(int64_t value, bool hex = false) {
    if (hex) {
        std::stringstream ss;
        ss << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
        return ss.str();
    }
    return detail::printf_format("%lld", value);
}

inline std::string format_u64(uint64_t value, bool hex = false) {
    if (hex) {
        std::stringstream ss;
        ss << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
        return ss.str();
    }
    return detail::printf_format("%lld", value);
}

// 格式化32位浮点数
inline std::string format_f32(float value) {
    return detail::printf_format("%f", value);
}

// 格式化64位浮点数
inline std::string format_f64(double value) {
    return detail::printf_format("%f", value);
}

// 格式化谓词值
inline std::string format_pred(bool value) { return value ? "true" : "false"; }

// 格式化内存地址
inline std::string format_address(uint64_t addr) {
    std::stringstream ss;
    ss << "0x" << std::hex << addr;
    return ss.str();
}

// 格式化寄存器值
template <typename T>
inline std::string format_register_value(T value, bool hex = false) {
    if constexpr (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else if constexpr (std::is_floating_point_v<T>) {
        return detail::printf_format("%.6f", value);
    } else if constexpr (std::is_integral_v<T> && sizeof(T) <= 4) {
        if (hex) {
            std::stringstream ss;
            ss << "0x" << std::hex << value;
            return ss.str();
        } else {
            return detail::printf_format("%d", static_cast<int>(value));
        }
    } else if constexpr (std::is_integral_v<T> && sizeof(T) > 4) {
        if (hex) {
            std::stringstream ss;
            ss << "0x" << std::hex << value;
            return ss.str();
        } else {
            return detail::printf_format("%lld", static_cast<long long>(value));
        }
    } else {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }
}

// 专门处理std::any类型的寄存器值格式化
inline std::string format_register_value(const std::any &value,
                                         bool hex = false) {
    if (!value.has_value()) {
        return "null";
    }

    try {
        // 尝试各种常见类型
        if (value.type() == typeid(bool)) {
            return format_register_value(any_cast<bool>(value), hex);
        } else if (value.type() == typeid(int8_t)) {
            return format_register_value(any_cast<int8_t>(value), hex);
        } else if (value.type() == typeid(uint8_t)) {
            return format_register_value(any_cast<uint8_t>(value), hex);
        } else if (value.type() == typeid(int16_t)) {
            return format_register_value(any_cast<int16_t>(value), hex);
        } else if (value.type() == typeid(uint16_t)) {
            return format_register_value(any_cast<uint16_t>(value), hex);
        } else if (value.type() == typeid(int32_t)) {
            return format_register_value(any_cast<int32_t>(value), hex);
        } else if (value.type() == typeid(uint32_t)) {
            return format_register_value(any_cast<uint32_t>(value), hex);
        } else if (value.type() == typeid(int64_t)) {
            return format_register_value(any_cast<int64_t>(value), hex);
        } else if (value.type() == typeid(uint64_t)) {
            return format_register_value(any_cast<uint64_t>(value), hex);
        } else if (value.type() == typeid(float)) {
            return format_register_value(any_cast<float>(value), hex);
        } else if (value.type() == typeid(double)) {
            return format_register_value(any_cast<double>(value), hex);
        } else {
            return "unknown_type";
        }
    } catch (...) {
        return "format_error";
    }
}

} // namespace debug_format

// 添加PTX_INFO_THREAD宏定义，确保在PTXDebugger类中可以使用
#define PTX_INFO_THREAD(fmt, ...)                                              \
    PTX_LOG_SIMPLE(ptxsim::log_level::info, "thread", fmt, ##__VA_ARGS__)

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

// 调试宏
#define PTX_TRACE_INSTR(pc, opcode, operands, blockIdx, threadIdx)             \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::trace,   \
                                                   "exec")) {                  \
            ptxsim::PTXDebugger::get().trace_instruction(pc, opcode, operands, \
                                                         blockIdx, threadIdx); \
        }                                                                      \
    } while (0)

#define PTX_TRACE_MEM_ACCESS(is_write, addr_expr, addr, size, value, blockIdx, \
                             threadIdx)                                        \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::trace,   \
                                                   "mem")) {                   \
            ptxsim::PTXDebugger::get().trace_memory_access(                    \
                is_write, addr_expr, addr, size, value, blockIdx, threadIdx);  \
        }                                                                      \
    } while (0)

#define PTX_TRACE_REG_ACCESS(reg_name, value, is_write, blockIdx, threadIdx)   \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::trace,   \
                                                   "reg")) {                   \
            ptxsim::PTXDebugger::get().trace_register_access(                  \
                reg_name, value, is_write, blockIdx, threadIdx);               \
        }                                                                      \
    } while (0)

#define PTX_CHECK_BREAKPOINT(pc, context)                                      \
    (ptxsim::PTXDebugger::get().check_breakpoint(pc, context))

// 在ptxsim命名空间内定义PTX_INFO_THREAD宏
#define PTX_INFO_THREAD(fmt, ...)                                              \
    PTX_LOG_SIMPLE(ptxsim::log_level::info, "thread", fmt, ##__VA_ARGS__)

// 添加缺失的PTX_INFO_THREAD宏定义
#define PTX_TRACE_EXEC(...)                                                    \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,    \
                                                   "exec")) {                  \
            ptxsim::Logger::log(ptxsim::log_level::info, "exec", __VA_ARGS__); \
        }                                                                      \
    } while (0)

#define PTX_TRACE_MEM(...)                                                     \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,    \
                                                   "mem")) {                   \
            ptxsim::Logger::log(ptxsim::log_level::info, "mem", __VA_ARGS__);  \
        }                                                                      \
    } while (0)

#define PTX_TRACE_REG(...)                                                     \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,    \
                                                   "reg")) {                   \
            ptxsim::Logger::log(ptxsim::log_level::info, "reg", __VA_ARGS__);  \
        }                                                                      \
    } while (0)

#define PTX_DUMP_THREAD_STATE(prefix, thread_ctx, block, thread)               \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,    \
                                                   "thread")) {                \
            ptxsim::PTXDebugger::get().dump_thread_state(prefix, thread_ctx,   \
                                                         block, thread);       \
        }                                                                      \
    } while (0)

// 性能计时器
class PerfTimer {
public:
    PerfTimer(const std::string &name, bool enabled = true)
        : name_(name), enabled_(enabled),
          start_(std::chrono::high_resolution_clock::now()) {
        if (enabled_) {
            PTX_DEBUG_EMU("Starting timer: %s", name_.c_str());
        }
    }

    ~PerfTimer() {
        if (enabled_) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start_);
            PTX_DEBUG_EMU("Timer %s took %ld microseconds", name_.c_str(),
                          duration.count());
        }
    }

private:
    std::string name_;
    bool enabled_;
    std::chrono::high_resolution_clock::time_point start_;
};

#define PTX_PERF_TIMER(name)                                                   \
    ptxsim::PerfTimer _perf_timer_(                                            \
        name, ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,  \
                                                     "perf"))

// 条件编译性能计时器
#ifdef PTXSIM_DISABLE_PERF_TIMING
#define PTX_PERF_TIMER_IF(cond, name) ((void)0)
#else
#define PTX_PERF_TIMER_IF(cond, name)                                          \
    ptxsim::PerfTimer _perf_timer_##__LINE__(                                  \
        name, (cond) && ptxsim::LoggerConfig::get().is_enabled(                \
                            ptxsim::log_level::info, "perf"))
#endif

} // namespace ptxsim

#endif // PTX_DEBUG_H