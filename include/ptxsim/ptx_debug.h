#ifndef PTX_DEBUG_H
#define PTX_DEBUG_H

#include "../utils/logger.h"
#include <any>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

struct StatementContext;
// 前向声明
// enum class StatementType;

namespace ptxsim {

// PTX指令类型
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
        std::lock_guard<std::mutex> lock(get_mutex());
        auto it = instruction_tracing_.find(type);
        return it != instruction_tracing_.end() && it->second;
    }

    // 启用/禁用内存访问跟踪
    void enable_memory_trace(bool enable = true) {
        std::lock_guard<std::mutex> lock(get_mutex());
        trace_memory_ = enable;
    }

    bool is_memory_traced() const {
        std::lock_guard<std::mutex> lock(get_mutex());
        return trace_memory_;
    }

    // 启用/禁用寄存器访问跟踪
    void enable_register_trace(bool enable = true) {
        std::lock_guard<std::mutex> lock(get_mutex());
        trace_registers_ = enable;
    }

    bool is_register_traced() const {
        std::lock_guard<std::mutex> lock(get_mutex());
        return trace_registers_;
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

    void clear_all_breakpoints() {
        std::lock_guard<std::mutex> lock(get_mutex());
        breakpoints_.clear();
    }

    bool has_breakpoint(
        int pc,
        const std::unordered_map<std::string, std::any> &context = {}) const {
        std::lock_guard<std::mutex> lock(get_mutex());
        auto it = breakpoints_.find(pc);
        if (it == breakpoints_.end())
            return false;

        // 检查条件
        if (it->second) {
            return it->second(pc, context);
        }
        return true;
    }

    // 设置观察点（寄存器）
    void set_watchpoint(const std::string &reg_name) {
        std::lock_guard<std::mutex> lock(get_mutex());
        watchpoints_.insert(reg_name);
    }

    void clear_watchpoint(const std::string &reg_name) {
        std::lock_guard<std::mutex> lock(get_mutex());
        watchpoints_.erase(reg_name);
    }

    void clear_all_watchpoints() {
        std::lock_guard<std::mutex> lock(get_mutex());
        watchpoints_.clear();
    }

    bool has_watchpoint(const std::string &reg_name) const {
        std::lock_guard<std::mutex> lock(get_mutex());
        return watchpoints_.find(reg_name) != watchpoints_.end();
    }

    // 设置PC范围
    void set_pc_range(int start, int end) {
        std::lock_guard<std::mutex> lock(get_mutex());
        pc_start_ = start;
        pc_end_ = end;
    }

    void clear_pc_range() {
        std::lock_guard<std::mutex> lock(get_mutex());
        pc_start_ = -1;
        pc_end_ = -1;
    }

    bool is_pc_traced(int pc) const {
        std::lock_guard<std::mutex> lock(get_mutex());
        if (pc_start_ == -1 || pc_end_ == -1)
            return true;
        return pc >= pc_start_ && pc <= pc_end_;
    }

    // 设置跟踪回调
    void set_instruction_callback(
        std::function<void(int, const std::string &)> callback) {
        std::lock_guard<std::mutex> lock(get_mutex());
        instruction_callback_ = callback;
    }

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
            std::ifstream config_file(filename);
            if (!config_file.is_open()) {
                return false;
            }

            std::string line;
            while (std::getline(config_file, line)) {
                // 跳过注释和空行
                if (line.empty() || line[0] == '#')
                    continue;

                size_t eq_pos = line.find('=');
                if (eq_pos == std::string::npos)
                    continue;

                std::string key = line.substr(0, eq_pos);
                std::string value = line.substr(eq_pos + 1);

                // 去除前后空格
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                // 处理配置项
                if (key == "trace_instruction") {
                    // 启用所有类型的指令跟踪
                    if (value == "true" || value == "1") {
                        instruction_tracing_[InstructionType::MEMORY] = true;
                        instruction_tracing_[InstructionType::ARITHMETIC] =
                            true;
                        instruction_tracing_[InstructionType::CONTROL] = true;
                        instruction_tracing_[InstructionType::LOGIC] = true;
                        instruction_tracing_[InstructionType::CONVERT] = true;
                        instruction_tracing_[InstructionType::SPECIAL] = true;
                        instruction_tracing_[InstructionType::OTHER] = true;
                    }
                } else if (key == "trace_memory") {
                    trace_memory_ = (value == "true" || value == "1");
                } else if (key == "trace_registers") {
                    trace_registers_ = (value == "true" || value == "1");
                } else if (key == "trace_pc_range" || key == "pc_range") {
                    size_t dash_pos = value.find('-');
                    if (dash_pos != std::string::npos) {
                        int start = std::stoi(value.substr(0, dash_pos));
                        int end = std::stoi(value.substr(dash_pos + 1));
                        pc_start_ = start;
                        pc_end_ = end;
                    }
                } else if (key.find("breakpoint.") == 0) {
                    int pc = std::stoi(key.substr(11));
                    breakpoints_[pc] = nullptr; // 无条件断点
                } else if (key.find("watchpoint.") == 0) {
                    std::string reg_name = value;
                    watchpoints_.insert(reg_name);
                } else if (key.find("trace_instruction_type.") == 0) {
                    std::string type_str = key.substr(24); // Fixed length
                    bool enable = (value == "true" || value == "1");

                    InstructionType type;
                    if (type_str == "memory")
                        type = InstructionType::MEMORY;
                    else if (type_str == "arithmetic")
                        type = InstructionType::ARITHMETIC;
                    else if (type_str == "control")
                        type = InstructionType::CONTROL;
                    else if (type_str == "logic")
                        type = InstructionType::LOGIC;
                    else if (type_str == "convert")
                        type = InstructionType::CONVERT;
                    else if (type_str == "special")
                        type = InstructionType::SPECIAL;
                    else if (type_str == "other")
                        type = InstructionType::OTHER;
                    else
                        continue;

                    instruction_tracing_[type] = enable;
                }
            }
            return true;
        } catch (...) {
            return false;
        }
    }

private:
    // 使用函数局部静态变量来避免死锁问题
    static std::mutex &get_mutex() {
        static std::mutex mutex_instance;
        return mutex_instance;
    }

    DebugConfig()
        : trace_memory_(false), trace_registers_(false), pc_start_(-1),
          pc_end_(-1) {
        // 默认启用控制流指令跟踪
        instruction_tracing_[InstructionType::CONTROL] = true;
    }

    std::unordered_map<InstructionType, bool> instruction_tracing_;
    bool trace_memory_;
    bool trace_registers_;
    std::unordered_map<int, BreakpointCondition> breakpoints_;
    std::unordered_set<std::string> watchpoints_;
    int pc_start_;
    int pc_end_;
    std::function<void(int, const std::string &)> instruction_callback_;

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

// 格式化线程/块坐标
inline std::string format_coord(int x, int y, int z) {
    return detail::printf_format("(%d,%d,%d)", x, y, z);
}

// 格式化寄存器值
inline std::string format_register_value(const std::any &value) {
    try {
        if (value.type() == typeid(int32_t)) {
            return format_i32(std::any_cast<int32_t>(value));
        } else if (value.type() == typeid(uint32_t)) {
            return format_u32(std::any_cast<uint32_t>(value));
        } else if (value.type() == typeid(int64_t)) {
            return format_i64(std::any_cast<int64_t>(value));
        } else if (value.type() == typeid(float)) {
            return format_f32(std::any_cast<float>(value));
        } else if (value.type() == typeid(double)) {
            return format_f64(std::any_cast<double>(value));
        } else if (value.type() == typeid(bool)) {
            return format_pred(std::any_cast<bool>(value));
        } else {
            return "[unknown type]";
        }
    } catch (const std::bad_any_cast &) {
        return "[bad cast]";
    } catch (...) {
        return "[error]";
    }
}
} // namespace debug_format

// 调试工具类
class PTXDebugger {
public:
    static PTXDebugger &get() {
        static PTXDebugger instance;
        return instance;
    }

    // 记录指令执行
    void trace_instruction(int pc, const std::string &opcode,
                           const std::string &operands, int block_x = 0,
                           int block_y = 0, int block_z = 0, int thread_x = 0,
                           int thread_y = 0, int thread_z = 0) {
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

        // 触发回调
        config.trigger_instruction_callback(pc, full_instruction);

        // 记录日志，包含线程和块索引信息
        PTX_TRACE_EXEC("[%4d] [%d,%d,%d][%d,%d,%d] %s", pc, block_x, block_y,
                       block_z, thread_x, thread_y, thread_z,
                       full_instruction.c_str());
    }

    // 记录内存访问
    void trace_memory_access(bool is_write, const std::string &addr_expr,
                             uint64_t addr, size_t size,
                             void *value = nullptr) {
        if (!ptxsim::DebugConfig::get().is_memory_traced())
            return;

        const char *access_type = is_write ? "WRITE" : "READ";
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
                value_str = ptxsim::debug_format::format_i64(u64, true) +
                            " / " + ptxsim::debug_format::format_f64(f64);
            } else {
                value_str =
                    ptxsim::detail::printf_format("0x%p", (void *)value);
            }

            PTX_TRACE_MEM("%s [%s] = %s (size=%zu)", access_type,
                          addr_str.c_str(), value_str.c_str(), size);
        } else {
            PTX_TRACE_MEM("%s [%s] (size=%zu)", access_type, addr_str.c_str(),
                          size);
        }
    }

    // 记录寄存器访问
    void trace_register_access(const std::string &reg_name,
                               const std::any &value, bool is_write) {
        if (!ptxsim::DebugConfig::get().is_register_traced())
            return;
        if (!ptxsim::DebugConfig::get().has_watchpoint(reg_name))
            return;

        const char *access_type = is_write ? "WRITE" : "READ";
        std::string value_str =
            ptxsim::debug_format::format_register_value(value);

        PTX_TRACE_REG("%s %s = %s", access_type, reg_name.c_str(),
                      value_str.c_str());
    }

    // 线程状态转储
    template <typename T>
    void dump_thread_state(const std::string &name, const T &state, int block_x,
                           int block_y, int block_z, int thread_x, int thread_y,
                           int thread_z) {
        std::stringstream ss;
        ss << "=== Thread State Dump ===" << std::endl;
        ss << "Thread: " << name << std::endl;
        ss << "Block: "
           << ptxsim::debug_format::format_coord(block_x, block_y, block_z)
           << std::endl;
        ss << "Thread: "
           << ptxsim::debug_format::format_coord(thread_x, thread_y, thread_z)
           << std::endl;
        ss << "PC: " << state.pc << " | State: "
           << (state.state == 0   ? "RUN"
               : state.state == 1 ? "EXIT"
                                  : "UNKNOWN")
           << std::endl;

        // 调用具体类型的转储函数
        state.dump_state(ss);

        ss << "=========================" << std::endl;

        PTX_DEBUG_THREAD("\n%s", ss.str().c_str());
        PTX_DEBUG_EMU_SIMPLE("\n%s", ss.str().c_str());
    }

    // 检查断点
    bool check_breakpoint(
        int pc, const std::unordered_map<std::string, std::any> &context = {}) {
        if (ptxsim::DebugConfig::get().has_breakpoint(pc, context)) {
            PTX_INFO_EMU("Hit breakpoint at PC=%d", pc);
            return true;
        }
        return false;
    }

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
            ptxsim::PTXDebugger::get().trace_instruction(                      \
                pc, opcode, operands, blockIdx.x, blockIdx.y, blockIdx.z,      \
                threadIdx.x, threadIdx.y, threadIdx.z);                        \
        }                                                                      \
    } while (0)

#define PTX_TRACE_MEM_ACCESS(is_write, addr_expr, addr, size, value)           \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::trace,   \
                                                   "mem")) {                   \
            ptxsim::PTXDebugger::get().trace_memory_access(                    \
                is_write, addr_expr, addr, size, value);                       \
        }                                                                      \
    } while (0)

#define PTX_TRACE_REG_ACCESS(reg_name, value, is_write)                        \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::trace,   \
                                                   "reg")) {                   \
            ptxsim::PTXDebugger::get().trace_register_access(reg_name, value,  \
                                                             is_write);        \
        }                                                                      \
    } while (0)

#define PTX_CHECK_BREAKPOINT(pc, context)                                      \
    (ptxsim::PTXDebugger::get().check_breakpoint(pc, context))

#define PTX_DUMP_THREAD_STATE(name, state, blockIdx, threadIdx)                \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::debug,   \
                                                   "thread")) {                \
            ptxsim::PTXDebugger::get().dump_thread_state(                      \
                name, state, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,  \
                threadIdx.y, threadIdx.z);                                     \
        }                                                                      \
    } while (0)

// 作用域性能计时器
class PerfTimer {
public:
    explicit PerfTimer(const std::string &name, bool enabled = true)
        : name_(name), enabled_(enabled) {
        if (enabled_) {
            start_ = std::chrono::high_resolution_clock::now();
        }
    }

    ~PerfTimer() {
        if (enabled_) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start_);

            PTX_INFO_EMU_SIMPLE("PERF[%s]: %s took %lld μs",
                                ptxsim::detail::current_thread_id().c_str(),
                                name_.c_str(),
                                static_cast<long long>(duration.count()));
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