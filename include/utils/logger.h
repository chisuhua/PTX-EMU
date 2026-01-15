// utils/logger.h
#ifndef PTX_LOGGER_H
#define PTX_LOGGER_H

#include "inipp/inipp.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip> // 用于时间格式化
#include <iostream>
#include <map> // 添加缺失的map头文件
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <source_location>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unordered_map>
#include <vector>

// 前向声明GPUContext类，避免包含完整头文件
class GPUContext;

// 获取GPU时钟数
std::string get_gpu_clock_str();
namespace ptxsim {

// ===========================================================================
// 日志级别定义
// ===========================================================================

enum class log_level {
    trace = 0,
    debug = 1,
    info = 2,
    warning = 3,
    error = 4,
    fatal = 5
};

// ===========================================================================
// 日志目标
// ===========================================================================

enum class log_target {
    console = 0, // 控制台
    file = 1,    // 文件
    both = 2     // 同时输出到控制台和文件
};

// ===========================================================================
// 日志格式选项
// ===========================================================================

struct LogFormatOptions {
    bool show_timestamp = true;
    bool show_level = true;
    bool show_component = true;
    bool show_location = true;
    bool show_thread_id = true;
    bool colorize = true; // 为不同级别添加颜色
};

// ===========================================================================
// printf 风格的字符串格式化工具
// ===========================================================================

namespace detail {
// 日志级别到字符串的转换
inline const char *log_level_str(log_level level) {
    switch (level) {
    case log_level::trace:
        return "[TRACE]";
    case log_level::debug:
        return "[DEBUG]";
    case log_level::info:
        return "[INFO]";
    case log_level::warning:
        return "[WARN]";
    case log_level::error:
        return "[ERROR]";
    case log_level::fatal:
        return "[FATAL]";
    default:
        return "[UNKNOWN]";
    }
}

// 获取当前时间戳
inline std::string current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// 获取当前线程ID
inline std::string current_thread_id() {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    return oss.str();
}

// ANSI颜色代码
inline const char *level_color(log_level level) {
    switch (level) {
    case log_level::trace:
        return "\033[36m"; // 青色
    case log_level::debug:
        return "\033[34m"; // 蓝色
    case log_level::info:
        return "\033[32m"; // 绿色
    case log_level::warning:
        return "\033[33m"; // 黄色
    case log_level::error:
        return "\033[31m"; // 红色
    case log_level::fatal:
        return "\033[35m"; // 紫色
    default:
        return "";
    }
}

// 重置颜色
inline const char *reset_color() { return "\033[0m"; }

// 输出日志的核心函数声明
void output_log(log_level level, const std::string &component,
                const std::string &msg, const std::source_location &loc);

// 输出简单日志的核心函数声明
void output_log_simple(log_level level, const std::string &component,
                       const std::string &msg);

// 格式化函数模板
template <typename... Args>
std::string printf_format(const char *fmt, Args &&...args) {
    if constexpr (sizeof...(Args) == 0) {
        return fmt ? std::string(fmt) : std::string();
    } else {
        if (!fmt)
            return std::string();
        size_t size = snprintf(nullptr, 0, fmt, args...) + 1;
        std::unique_ptr<char[]> buf(new char[size]);
        snprintf(buf.get(), size, fmt, args...);
        return std::string(buf.get(), buf.get() + size - 1);
    }
}
} // namespace detail

// ===========================================================================
// 异步日志队列
// ===========================================================================

class AsyncLogQueue {
public:
    struct LogEntry {
        log_level level;
        std::string component;
        std::string message;
        std::string timestamp;
        std::string thread_id;
        std::source_location loc;
    };

    AsyncLogQueue() : stop_(false) {
        worker_thread_ = std::thread(&AsyncLogQueue::process_queue, this);
    }

    ~AsyncLogQueue() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            cv_.notify_one();
        }
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    void enqueue(LogEntry entry) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(entry));
        cv_.notify_one();
    }

    void set_output_callback(std::function<void(const LogEntry &)> callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        output_callback_ = callback;
    }

private:
    void process_queue() {
        while (true) {
            LogEntry entry;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return !queue_.empty() || stop_; });

                if (stop_ && queue_.empty()) {
                    return;
                }

                entry = std::move(queue_.front());
                queue_.pop();
            }

            if (output_callback_) {
                output_callback_(entry);
            }
        }
    }

    std::queue<LogEntry> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    std::atomic<bool> stop_;
    std::function<void(const LogEntry &)> output_callback_;
};

// ===========================================================================
// 全局日志配置
// ===========================================================================

class LoggerConfig {
public:
    // 获取单例实例
    static LoggerConfig &get() {
        static LoggerConfig instance;
        return instance;
    }

    // 构造函数
    LoggerConfig();

    // 从INI配置部分加载日志配置
    void load_from_ini_section(const inipp::Ini<char>::Section &logger_section);

    struct FormatOptions {
        bool show_timestamp = true;
        bool show_level = true;
        bool show_component = true;
        bool show_location = true;
        bool show_thread_id = true;
        bool colorize = true;
    };

    // 新增：获取格式选项
    const FormatOptions &get_format_options() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return format_options_;
    }

    // 新增：检查指定级别和组件的日志是否启用
    bool is_enabled(log_level level, const std::string &component) const {
        std::lock_guard<std::mutex> lock(mutex_);
        log_level threshold = global_level_;
        auto it = component_levels_.find(component);
        if (it != component_levels_.end()) {
            threshold = it->second;
        }
        return level >= threshold;
    }

    // Getter和Setter方法
    log_level get_global_level() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return global_level_;
    }

    void set_global_level(log_level level) {
        std::lock_guard<std::mutex> lock(mutex_);
        global_level_ = level;
    }

    log_target get_target() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return target_;
    }

    void set_target(log_target target) {
        std::lock_guard<std::mutex> lock(mutex_);
        target_ = target;
    }

    void set_target_from_string(const std::string &target_str) {
        if (target_str == "console") {
            set_target(log_target::console);
        } else if (target_str == "file") {
            set_target(log_target::file);
        } else if (target_str == "both") {
            set_target(log_target::both);
        } else {
            set_target(log_target::console); // 默认值
        }
    }

    bool is_async_logging_enabled() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return use_async_logging_;
    }

    void enable_async_logging(bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);
        use_async_logging_ = enable;
    }

    bool get_use_color_output() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return use_color_output_;
    }

    void set_use_color_output(bool colorize) {
        std::lock_guard<std::mutex> lock(mutex_);
        use_color_output_ = colorize;
    }

    // 设置日志文件
    void set_logfile(const std::string &path) {
        std::lock_guard<std::mutex> lock(mutex_);
        logfile_path_ = path;
        set_logfile_internal(path);
    }

    const std::string &get_logfile_path() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return logfile_path_;
    }

    // 组件级别日志控制
    log_level get_component_level(const std::string &component) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = component_levels_.find(component);
        if (it != component_levels_.end()) {
            return it->second;
        }
        return global_level_; // 默认返回全局级别
    }

    void set_component_level(const std::string &component, log_level level) {
        std::lock_guard<std::mutex> lock(mutex_);
        component_levels_[component] = level;
    }

    // 工具方法
    log_level string_to_log_level(const std::string &level_str) const {
        if (level_str == "trace") {
            return log_level::trace;
        } else if (level_str == "debug") {
            return log_level::debug;
        } else if (level_str == "info") {
            return log_level::info;
        } else if (level_str == "warning") {
            return log_level::warning;
        } else if (level_str == "error") {
            return log_level::error;
        } else if (level_str == "fatal") {
            return log_level::fatal;
        } else {
            return log_level::info; // 默认值
        }
    }

    // 添加一个公共方法用于写入日志文件
    void write_to_logfile(const std::string &message) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (logfile_.is_open()) {
            logfile_ << message;
            logfile_.flush();
        }
    }

    // 添加一个公共方法用于检查日志文件是否打开
    bool is_logfile_open() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return logfile_.is_open();
    }

private:
    // 私有方法
    void set_logfile_internal(const std::string &path) {
        // 关闭现有文件
        if (logfile_.is_open()) {
            logfile_.close();
        }

        // 尝试打开新文件
        logfile_.open(path, std::ios_base::out | std::ios_base::trunc);
        if (!logfile_.is_open()) {
            // 如果无法打开文件，回退到控制台输出
            std::cerr << "Warning: Could not open log file: " << path
                      << ", falling back to console output." << std::endl;
        }
    }

    // std::string trim(const std::string &str) const {
    //     size_t start = str.find_first_not_of(" \t\r\n");
    //     if (start == std::string::npos) {
    //         return "";
    //     }
    //     size_t end = str.find_last_not_of(" \t\r\n");
    //     return str.substr(start, end - start + 1);
    // }

    // 成员变量
    log_level global_level_;
    log_target target_;
    mutable std::mutex mutex_;
    bool use_async_logging_;
    bool use_color_output_ = true;
    std::string logfile_path_;
    std::ofstream logfile_; // 使用ofstream而不是FILE*
    std::map<std::string, log_level> component_levels_;

    // 新增成员变量
    FormatOptions format_options_;
};
// ===========================================================================
// 基础日志输出函数
// ===========================================================================

namespace detail {
// 输出日志的核心函数
inline void output_log(log_level level, const std::string &component,
                       const std::string &message,
                       const std::source_location &loc) {
    // 函数实现保持不变
    auto &config = LoggerConfig::get();
    if (!config.is_enabled(level, component))
        return;

    // 异步记录
    if (config.get_target() != log_target::file &&
        config.get_format_options().colorize) {
        std::cout << detail::level_color(level);
    }

    std::stringstream ss;

    const auto &format_opts = config.get_format_options();

    if (format_opts.show_timestamp) {
        // 添加时间戳 - 现在打印GPU时钟
        ss << "[" << get_gpu_clock_str() << "] ";
    }

    if (format_opts.show_level) {
        ss << log_level_str(level) << " ";
    }

    if (format_opts.show_component) {
        ss << "[" << component << "] ";
    }

    if (format_opts.show_thread_id) {
        ss << "[TID:" << detail::current_thread_id() << "] ";
    }

    ss << message;

    if (config.get_target() != log_target::file && format_opts.show_location &&
        loc.file_name()[0]) {
        ss << " at " << loc.file_name() << ":" << loc.line();
    }

    std::string formatted_message = ss.str();

    if (config.get_target() != log_target::file) {
        // 控制台输出
        switch (level) {
        case log_level::trace:
        case log_level::debug:
        case log_level::info:
            std::cout << formatted_message << std::endl;
            break;
        case log_level::warning:
        case log_level::error:
        case log_level::fatal:
            std::cerr << formatted_message << std::endl;
            break;
        }

        if (config.get_format_options().colorize) {
            std::cout << detail::reset_color();
            std::cerr << detail::reset_color();
        }
    }

    if (config.get_target() != log_target::console) {
        // 文件输出
        auto &config_ref = LoggerConfig::get();
        std::string full_message = formatted_message;
        if (format_opts.show_location && loc.file_name()[0]) {
            full_message += " at " + std::string(loc.file_name()) + ":" +
                            std::to_string(loc.line());
        }
        full_message += "\n";
        config_ref.write_to_logfile(full_message);
    }

    // 致命错误需要特殊处理
    if (level == log_level::fatal) {
        std::abort();
    }
}

// 简洁日志输出（不显示位置信息）
inline void output_log_simple(log_level level, const std::string &component,
                              const std::string &message) {
    auto &config = LoggerConfig::get();
    if (!config.is_enabled(level, component))
        return;

    std::stringstream ss;

    const auto &format_opts = config.get_format_options();

    if (format_opts.show_timestamp) {
        // 添加时间戳 - 现在打印GPU时钟
        ss << "[" << get_gpu_clock_str() << "] ";
    }

    if (format_opts.show_level) {
        ss << log_level_str(level) << " ";
    }

    if (format_opts.show_component) {
        ss << "[" << component << "] ";
    }

    if (format_opts.show_thread_id) {
        ss << "[TID:" << detail::current_thread_id() << "] ";
    }

    ss << message;

    std::string formatted_message = ss.str();

    if (config.get_target() != log_target::file &&
        config.get_format_options().colorize) {
        std::cout << detail::level_color(level);
    }

    if (config.get_target() != log_target::file) {
        // 控制台输出
        if (level <= log_level::info) {
            std::cout << formatted_message << std::endl;
        } else {
            std::cerr << formatted_message << std::endl;
        }

        if (config.get_format_options().colorize) {
            std::cout << detail::reset_color();
            std::cerr << detail::reset_color();
        }
    }

    if (config.get_target() != log_target::console) {
        // 文件输出
        auto &config_ref = LoggerConfig::get();
        std::string full_message = formatted_message + "\n";
        config_ref.write_to_logfile(full_message);
    }

    // 致命错误需要特殊处理
    if (level == log_level::fatal) {
        std::abort();
    }
}
} // namespace detail

// ===========================================================================
// 宏定义
// ===========================================================================
#ifdef LOG
#define LOG_FUNC() std::cout << __func__ << std::endl
#else
#define LOG_FUNC() ((void)0)
#endif

// 基础日志宏
#define PTX_LOG(level, component, fmt, ...)                                    \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(level, component)) {        \
            auto loc = std::source_location::current();                        \
            std::string msg =                                                  \
                ptxsim::detail::printf_format(fmt, ##__VA_ARGS__);             \
            ptxsim::detail::output_log(level, component, msg, loc);            \
        }                                                                      \
    } while (0)

// 简洁日志宏
#define PTX_LOG_SIMPLE(level, component, fmt, ...)                             \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(level, component)) {        \
            std::string msg =                                                  \
                ptxsim::detail::printf_format(fmt, ##__VA_ARGS__);             \
            ptxsim::detail::output_log_simple(level, component, msg);          \
        }                                                                      \
    } while (0)

// 可切换版本的日志宏，默认使用PTX_LOG_SIMPLE
#ifdef PTXSIM_USE_DETAILED_LOGGING
#define PTX_DEBUG_EMU(fmt, ...)                                                \
    PTX_LOG(ptxsim::log_level::debug, "emu", fmt, ##__VA_ARGS__)
#define PTX_INFO_EMU(fmt, ...)                                                 \
    PTX_LOG(ptxsim::log_level::info, "emu", fmt, ##__VA_ARGS__)
#define PTX_WARN_EMU(fmt, ...)                                                 \
    PTX_LOG(ptxsim::log_level::warning, "emu", fmt, ##__VA_ARGS__)
#define PTX_ERROR_EMU(fmt, ...)                                                \
    PTX_LOG(ptxsim::log_level::error, "emu", fmt, ##__VA_ARGS__)
#define PTX_FATAL_EMU(fmt, ...)                                                \
    PTX_LOG(ptxsim::log_level::fatal, "emu", fmt, ##__VA_ARGS__)

#define PTX_DEBUG_EXEC(fmt, ...)                                               \
    PTX_LOG(ptxsim::log_level::debug, "exec", fmt, ##__VA_ARGS__)
#define PTX_INFO_EXEC(fmt, ...)                                                \
    PTX_LOG(ptxsim::log_level::info, "exec", fmt, ##__VA_ARGS__)

#define PTX_DEBUG_REG(fmt, ...)                                                \
    PTX_LOG(ptxsim::log_level::debug, "reg", fmt, ##__VA_ARGS__)

#define PTX_DEBUG_MEM(fmt, ...)                                                \
    PTX_LOG(ptxsim::log_level::debug, "mem", fmt, ##__VA_ARGS__)

#define PTX_DEBUG_THREAD(fmt, ...)                                             \
    PTX_LOG(ptxsim::log_level::debug, "thread", fmt, ##__VA_ARGS__)

#define PTX_DEBUG_EMU_SIMPLE(fmt, ...)                                         \
    PTX_LOG(ptxsim::log_level::debug, "emu", fmt, ##__VA_ARGS__)
#define PTX_INFO_EMU_SIMPLE(fmt, ...)                                          \
    PTX_LOG(ptxsim::log_level::info, "emu_simple", fmt, ##__VA_ARGS__)
#else
#define PTX_DEBUG_EMU(fmt, ...)                                                \
    PTX_LOG_SIMPLE(ptxsim::log_level::debug, "emu", fmt, ##__VA_ARGS__)
#define PTX_INFO_EMU(fmt, ...)                                                 \
    PTX_LOG_SIMPLE(ptxsim::log_level::info, "emu", fmt, ##__VA_ARGS__)
#define PTX_WARN_EMU(fmt, ...)                                                 \
    PTX_LOG_SIMPLE(ptxsim::log_level::warning, "emu", fmt, ##__VA_ARGS__)
#define PTX_ERROR_EMU(fmt, ...)                                                \
    PTX_LOG_SIMPLE(ptxsim::log_level::error, "emu", fmt, ##__VA_ARGS__)
#define PTX_FATAL_EMU(fmt, ...)                                                \
    PTX_LOG_SIMPLE(ptxsim::log_level::fatal, "emu", fmt, ##__VA_ARGS__)

#define PTX_DEBUG_EXEC(fmt, ...)                                               \
    PTX_LOG_SIMPLE(ptxsim::log_level::debug, "exec", fmt, ##__VA_ARGS__)
#define PTX_INFO_EXEC(fmt, ...)                                                \
    PTX_LOG_SIMPLE(ptxsim::log_level::info, "exec", fmt, ##__VA_ARGS__)

#define PTX_DEBUG_REG(fmt, ...)                                                \
    PTX_LOG_SIMPLE(ptxsim::log_level::debug, "reg", fmt, ##__VA_ARGS__)

#define PTX_DEBUG_MEM(fmt, ...)                                                \
    PTX_LOG_SIMPLE(ptxsim::log_level::debug, "mem", fmt, ##__VA_ARGS__)

#define PTX_DEBUG_THREAD(fmt, ...)                                             \
    PTX_LOG_SIMPLE(ptxsim::log_level::debug, "thread", fmt, ##__VA_ARGS__)

#define PTX_DEBUG_EMU_SIMPLE(fmt, ...)                                         \
    PTX_LOG_SIMPLE(ptxsim::log_level::debug, "emu", fmt, ##__VA_ARGS__)
#define PTX_INFO_EMU_SIMPLE(fmt, ...)                                          \
    PTX_LOG_SIMPLE(ptxsim::log_level::info, "emu_simple", fmt, ##__VA_ARGS__)
#endif

// 条件编译日志宏
#ifdef PTXSIM_DISABLE_LOGGING
#define PTX_DEBUG_EMU_IF(cond, fmt, ...) ((void)0)
#else
#ifdef PTXSIM_USE_DETAILED_LOGGING

#define PTX_DEBUG_EMU_IF(cond, fmt, ...)                                       \
    do {                                                                       \
        if ((cond) && ptxsim::LoggerConfig::get().is_enabled(                  \
                          ptxsim::log_level::debug, "emu")) {                  \
            auto loc = std::source_location::current();                        \
            std::string msg =                                                  \
                ptxsim::detail::printf_format(fmt, ##__VA_ARGS__);             \
            ptxsim::detail::output_log(ptxsim::log_level::debug, "emu", msg,   \
                                       loc);                                   \
        }                                                                      \
    } while (0)
#else

#define PTX_DEBUG_EMU_IF(cond, fmt, ...)                                       \
    do {                                                                       \
        if ((cond) && ptxsim::LoggerConfig::get().is_enabled(                  \
                          ptxsim::log_level::debug, "emu")) {                  \
            std::string msg =                                                  \
                ptxsim::detail::printf_format(fmt, ##__VA_ARGS__);             \
            ptxsim::detail::output_log_simple(ptxsim::log_level::debug, "emu", \
                                              msg);                            \
        }                                                                      \
    } while (0)
#endif
#endif

// 变量跟踪宏
#define PTX_DEBUG_VAR(component, var)                                          \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::debug,   \
                                                   component)) {               \
            std::string var_value = ptxsim::detail::to_string(var);            \
            ptxsim::detail::output_log_simple(                                 \
                ptxsim::log_level::debug, component,                           \
                ptxsim::detail::printf_format("%s = %s", #var,                 \
                                              var_value.c_str()));             \
        }                                                                      \
    } while (0)

// 指针跟踪宏
#define PTX_DEBUG_PTR(component, ptr)                                          \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::debug,   \
                                                   component)) {               \
            ptxsim::detail::output_log_simple(                                 \
                ptxsim::log_level::debug, component,                           \
                ptxsim::detail::printf_format(                                 \
                    "%s = 0x%llx", #ptr,                                       \
                    (unsigned long long)(uintptr_t)(ptr)));                    \
        }                                                                      \
    } while (0)

// 条件检查宏
#define PTX_CHECK(condition, component, fmt, ...)                              \
    do {                                                                       \
        if (!(condition) && ptxsim::LoggerConfig::get().is_enabled(            \
                                ptxsim::log_level::error, component)) {        \
            auto loc = std::source_location::current();                        \
            std::string msg = ptxsim::detail::printf_format(                   \
                "Check failed [%s]: " fmt, #condition, ##__VA_ARGS__);         \
            ptxsim::detail::output_log(ptxsim::log_level::error, component,    \
                                       msg, loc);                              \
        }                                                                      \
    } while (0)

// 致命错误宏
#ifdef PTXSIM_USE_DETAILED_LOGGING
#define PTX_FATAL(component, fmt, ...)                                         \
    do {                                                                       \
        auto loc = std::source_location::current();                            \
        std::string msg =                                                      \
            ptxsim::detail::printf_format("FATAL: " fmt, ##__VA_ARGS__);       \
        ptxsim::detail::output_log(ptxsim::log_level::fatal, component, msg,   \
                                   loc);                                       \
    } while (0)
#else
#define PTX_FATAL(component, fmt, ...)                                         \
    do {                                                                       \
        auto loc = std::source_location::current();                            \
        std::string msg =                                                      \
            ptxsim::detail::printf_format("FATAL: " fmt, ##__VA_ARGS__);       \
        ptxsim::detail::output_log_simple(ptxsim::log_level::fatal, component, \
                                          msg);                                \
    } while (0)
#endif

// 作用域退出
class ScopeExit {
public:
    explicit ScopeExit(std::function<void()> func) : func_(std::move(func)) {}

    ScopeExit(const ScopeExit &) = delete;
    ScopeExit &operator=(const ScopeExit &) = delete;
    ScopeExit(ScopeExit &&) = default;
    ScopeExit &operator=(ScopeExit &&) = default;

    ~ScopeExit() {
        if (func_) {
            func_();
        }
    }

    static void *operator new(std::size_t) = delete;
    static void *operator new[](std::size_t) = delete;

private:
    std::function<void()> func_;
};

#define PTX_SCOPE_EXIT(func) ptxsim::ScopeExit _scope_exit_([&]() { func; })

// 杂项工具
template <typename... Args> void unused([[maybe_unused]] Args &&...args) {}

#define PTX_UNUSED(...) ptxsim::unused(__VA_ARGS__)

// 输出格式化日志的辅助函数
template <typename... Args>
void printf_to_logger(log_level level, const std::string &component,
                      const char *fmt, Args &&...args) {
    auto formatted_msg =
        detail::printf_format(fmt, std::forward<Args>(args)...);
    detail::output_log(level, component, formatted_msg,
                       std::source_location::current());
}

// 输出简单格式化日志的辅助函数
template <typename... Args>
void printf_to_logger_simple(log_level level, const std::string &component,
                             const char *fmt, Args &&...args) {
    auto formatted_msg =
        detail::printf_format(fmt, std::forward<Args>(args)...);
    detail::output_log_simple(level, component, formatted_msg);
}

} // namespace ptxsim

#endif // PTX_LOGGER_H