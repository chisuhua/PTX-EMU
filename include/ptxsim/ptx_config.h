#ifndef PTX_CONFIG_H
#define PTX_CONFIG_H

#include "inipp/inipp.h"
#include <any>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ptx_ir/statement_context.h"
#include "utils/logger.h"

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
    static std::string getOperandsString(const StatementContext &statement);
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

    // 从INI section加载调试器配置 - 供内部使用
    void
    load_from_ini_section(const inipp::Ini<char>::Section &debugger_section);

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
                                    const std::string &key,
                                    InstructionType type) {
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
            instruction_map = {// 内存操作
                               {"ld", InstructionType::MEMORY},
                               {"st", InstructionType::MEMORY},
                               {"ldu", InstructionType::MEMORY},

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

} // namespace ptxsim

#endif // PTX_CONFIG_H