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
#include <iostream>
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

namespace ptxsim {

// ===========================================================================
// 日志级别定义
// ===========================================================================

enum class log_level {
    fatal = 0,
    error = 1,
    warning = 2,
    info = 3,
    trace = 4, // 比debug更详细，用于指令级跟踪
    debug = 5
};

// ===========================================================================
// 日志目标
// ===========================================================================

enum class log_target {
    console, // 控制台
    file,    // 文件
    both     // 同时输出到控制台和文件
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
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;
    auto timer = std::chrono::system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);
    char buffer[32];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &bt);
    sprintf(buffer + strlen(buffer), ".%03ld", ms.count());
    return std::string(buffer);
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
    // 添加检查以确保 fmt 是有效的格式字符串
    if (!fmt)
        return std::string();
    size_t size = snprintf(nullptr, 0, fmt, args...) + 1;
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, fmt, args...);
    return std::string(buf.get(), buf.get() + size - 1);
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
        std::source_location loc;
        std::chrono::system_clock::time_point timestamp;
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
    LoggerConfig()
        : global_level_(log_level::info), target_(log_target::console),
          use_async_logging_(false) {}

    // 从配置文件加载设置
    bool load_from_file(const std::string &filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            inipp::Ini<char> ini;
            std::ifstream is(filename);
            if (!is.is_open()) {
                return false;
            }

            ini.parse(is);
            ini.strip_trailing_comments(); // 移除值末尾的空格

            auto &section =
                ini.sections["logger"]; // 假设所有配置都在 [logger] 段下

            // 解析全局日志级别
            std::string level_str;
            inipp::get_value(section, "global_level", level_str);
            if (!level_str.empty()) {
                global_level_ = string_to_log_level(level_str);
            }

            // 解析日志目标
            std::string target_str;
            inipp::get_value(section, "target", target_str);
            if (!target_str.empty()) {
                set_target_from_string(target_str);
            }

            // 解析日志文件路径
            std::string logfile_str;
            inipp::get_value(section, "logfile", logfile_str);
            if (!logfile_str.empty()) {
                logfile_path_ = logfile_str;
                // 只有当路径设置成功时才尝试打开文件
                set_logfile_internal(logfile_str);
            }

            // 解析异步日志
            std::string async_str;
            inipp::get_value(section, "async", async_str);
            use_async_logging_ = (async_str == "true" || async_str == "1");

            // 解析颜色输出
            std::string colorize_str;
            inipp::get_value(section, "colorize", colorize_str);
            format_options_.colorize =
                (colorize_str == "true" || colorize_str == "1");

            // 解析组件级别配置 - 使用辅助函数减少重复
            parse_component_levels(section);

            return true;
        } catch (const std::exception &e) {
            // 可以考虑添加一个内部错误日志，但要避免递归调用
            return false;
        }
    }

    // 设置全局日志级别（字符串形式）
    void set_global_level_from_string(const std::string &level_str) {
        log_level level = global_level_;
        if (level_str == "trace")
            level = log_level::trace;
        else if (level_str == "debug")
            level = log_level::debug;
        else if (level_str == "info")
            level = log_level::info;
        else if (level_str == "warning")
            level = log_level::warning;
        else if (level_str == "error")
            level = log_level::error;
        else if (level_str == "fatal")
            level = log_level::fatal;

        // 注意：此处不再加锁，因为调用此函数的load_from_file已经持有锁
        global_level_ = level;
    }

    // 设置特定组件的日志级别（字符串形式）
    void set_component_level_from_string(const std::string &component,
                                         const std::string &level_str) {
        log_level level = log_level::info; // 默认级别
        if (level_str == "trace")
            level = log_level::trace;
        else if (level_str == "debug")
            level = log_level::debug;
        else if (level_str == "info")
            level = log_level::info;
        else if (level_str == "warning")
            level = log_level::warning;
        else if (level_str == "error")
            level = log_level::error;
        else if (level_str == "fatal")
            level = log_level::fatal;

        // 注意：此处不再加锁，因为调用此函数的load_from_file已经持有锁
        component_levels_[component] = level;
    }

    // 设置日志输出目标（字符串形式）
    void set_target_from_string(const std::string &target_str) {
        if (target_str == "console")
            target_ = log_target::console;
        else if (target_str == "file")
            target_ = log_target::file;
        else if (target_str == "both")
            target_ = log_target::both;
    }

    // 设置全局日志级别
    void set_global_level(log_level level) {
        std::lock_guard<std::mutex> lock(mutex_);
        global_level_ = level;
    }

    // 设置特定组件的日志级别
    void set_component_level(const std::string &component, log_level level) {
        std::lock_guard<std::mutex> lock(mutex_);
        component_levels_[component] = level;
    }

    // 获取组件的有效日志级别
    log_level get_effective_level(const std::string &component) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = component_levels_.find(component);
        if (it != component_levels_.end()) {
            return it->second;
        }
        return global_level_;
    }

    // 设置日志输出目标
    void set_target(log_target target) {
        std::lock_guard<std::mutex> lock(mutex_);
        target_ = target;
    }

    // 设置日志文件 (外部调用接口，带锁保护)
    bool set_logfile(const std::string &filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        return set_logfile_internal(filename);
    }

    // 设置日志文件 (内部使用，无锁)
    bool set_logfile_internal(const std::string &filename) {
        try {
            logfile_.open(filename, std::ios::app);
            if (!logfile_.is_open()) {
                return false;
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    // 检查是否启用了某个级别的日志
    bool is_enabled(log_level level, const std::string &component = "") const {
        return level >= get_effective_level(component);
    }

    // 获取全局日志级别
    log_level get_global_level() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return global_level_;
    }

    // 获取当前日志目标
    log_target get_target() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return target_;
    }

    // 获取日志文件路径
    const std::string &get_logfile_path() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return logfile_path_;
    }

    // 检查是否使用异步日志记录
    bool use_async_logging() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return use_async_logging_;
    }

    // 设置是否使用颜色输出
    void set_use_color_output(bool use_color) {
        std::lock_guard<std::mutex> lock(mutex_);
        format_options_.colorize = use_color;
    }

    // 获取是否使用颜色输出
    bool get_use_color_output() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return format_options_.colorize;
    }

    // 内部辅助方法：字符串转日志级别
    log_level string_to_log_level(const std::string &level_str) const {
        if (level_str == "trace")
            return log_level::trace;
        else if (level_str == "debug")
            return log_level::debug;
        else if (level_str == "info")
            return log_level::info;
        else if (level_str == "warning")
            return log_level::warning;
        else if (level_str == "error")
            return log_level::error;
        else if (level_str == "fatal")
            return log_level::fatal;
        else
            return log_level::info; // 默认值
    }

    // 禁用所有日志
    void disable_all() { set_global_level(log_level::fatal); }

    // 启用/禁用异步日志记录
    void enable_async_logging(bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);
        use_async_logging_ = enable;
    }

    // 获取日志格式选项
    const LogFormatOptions &get_format_options() const {
        return format_options_;
    }

    // 设置日志格式选项
    void set_format_options(const LogFormatOptions &options) {
        format_options_ = options;
    }

    ~LoggerConfig() {
        if (logfile_.is_open()) {
            logfile_.close();
        }
    }

    // 私有成员变量
private:
    mutable std::mutex mutex_;
    log_level global_level_;
    std::unordered_map<std::string, log_level> component_levels_;
    log_target target_;
    std::ofstream logfile_;
    std::string logfile_path_; // 存储日志文件路径
    bool use_async_logging_;
    LogFormatOptions format_options_;
    std::unique_ptr<AsyncLogQueue> async_queue_;

    // 禁止拷贝构造和赋值
    LoggerConfig(const LoggerConfig &) = delete;
    LoggerConfig &operator=(const LoggerConfig &) = delete;

    // 友元声明，允许detail命名空间中的函数访问私有成员
    friend void detail::output_log(log_level level,
                                   const std::string &component,
                                   const std::string &msg,
                                   const std::source_location &loc);
    friend void detail::output_log_simple(log_level level,
                                          const std::string &component,
                                          const std::string &msg);

    // 辅助函数：解析组件级别配置
    void parse_component_levels(const inipp::Ini<char>::Section &section) {
        for (const auto &pair : section) {
            const std::string &key = pair.first;
            const std::string &value = pair.second;
            if (key.rfind("component.", 0) == 0) { // 检查是否以 "component." 开头
                std::string component = key.substr(10); // 去掉 "component." 前缀
                set_component_level_from_string(component, value);
            }
        }
    }
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
        ss << "[" << detail::current_timestamp() << "] ";
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
        {
            std::lock_guard<std::mutex> lock(config_ref.mutex_);
            if (config_ref.logfile_.is_open()) {
                config_ref.logfile_ << formatted_message;
                if (format_opts.show_location && loc.file_name()[0]) {
                    config_ref.logfile_ << " at " << loc.file_name() << ":"
                                        << loc.line();
                }
                config_ref.logfile_ << std::endl;
                config_ref.logfile_.flush();
            }
        }
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
        ss << "[" << detail::current_timestamp() << "] ";
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
        {
            std::lock_guard<std::mutex> lock(config_ref.mutex_);
            if (config_ref.logfile_.is_open()) {
                config_ref.logfile_ << formatted_message << std::endl;
                config_ref.logfile_.flush();
            }
        }
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
#define PTX_TRACE_EMU(fmt, ...)                                                \
    PTX_LOG(ptxsim::log_level::trace, "emu", fmt, ##__VA_ARGS__)
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

#define PTX_TRACE_EXEC(fmt, ...)                                               \
    PTX_LOG(ptxsim::log_level::trace, "exec", fmt, ##__VA_ARGS__)
#define PTX_DEBUG_EXEC(fmt, ...)                                               \
    PTX_LOG(ptxsim::log_level::debug, "exec", fmt, ##__VA_ARGS__)
#define PTX_INFO_EXEC(fmt, ...)                                                \
    PTX_LOG(ptxsim::log_level::info, "exec", fmt, ##__VA_ARGS__)

#define PTX_TRACE_REG(fmt, ...)                                                \
    PTX_LOG(ptxsim::log_level::trace, "reg", fmt, ##__VA_ARGS__)
#define PTX_DEBUG_REG(fmt, ...)                                                \
    PTX_LOG(ptxsim::log_level::debug, "reg", fmt, ##__VA_ARGS__)

#define PTX_TRACE_MEM(fmt, ...)                                                \
    PTX_LOG(ptxsim::log_level::trace, "mem", fmt, ##__VA_ARGS__)
#define PTX_DEBUG_MEM(fmt, ...)                                                \
    PTX_LOG(ptxsim::log_level::debug, "mem", fmt, ##__VA_ARGS__)

#define PTX_TRACE_THREAD(fmt, ...)                                             \
    PTX_LOG(ptxsim::log_level::trace, "thread", fmt, ##__VA_ARGS__)
#define PTX_DEBUG_THREAD(fmt, ...)                                             \
    PTX_LOG(ptxsim::log_level::debug, "thread", fmt, ##__VA_ARGS__)

#define PTX_TRACE_EMU_SIMPLE(fmt, ...)                                         \
    PTX_LOG(ptxsim::log_level::trace, "emu", fmt, ##__VA_ARGS__)
#define PTX_DEBUG_EMU_SIMPLE(fmt, ...)                                         \
    PTX_LOG(ptxsim::log_level::debug, "emu", fmt, ##__VA_ARGS__)
#define PTX_INFO_EMU_SIMPLE(fmt, ...)                                          \
    PTX_LOG(ptxsim::log_level::info, "emu_simple", fmt, ##__VA_ARGS__)
#else
#define PTX_TRACE_EMU(fmt, ...)                                                \
    PTX_LOG_SIMPLE(ptxsim::log_level::trace, "emu", fmt, ##__VA_ARGS__)
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

#define PTX_TRACE_EXEC(fmt, ...)                                               \
    PTX_LOG_SIMPLE(ptxsim::log_level::trace, "exec", fmt, ##__VA_ARGS__)
#define PTX_DEBUG_EXEC(fmt, ...)                                               \
    PTX_LOG_SIMPLE(ptxsim::log_level::debug, "exec", fmt, ##__VA_ARGS__)
#define PTX_INFO_EXEC(fmt, ...)                                                \
    PTX_LOG_SIMPLE(ptxsim::log_level::info, "exec", fmt, ##__VA_ARGS__)

#define PTX_TRACE_REG(fmt, ...)                                                \
    PTX_LOG_SIMPLE(ptxsim::log_level::trace, "reg", fmt, ##__VA_ARGS__)
#define PTX_DEBUG_REG(fmt, ...)                                                \
    PTX_LOG_SIMPLE(ptxsim::log_level::debug, "reg", fmt, ##__VA_ARGS__)

#define PTX_TRACE_MEM(fmt, ...)                                                \
    PTX_LOG_SIMPLE(ptxsim::log_level::trace, "mem", fmt, ##__VA_ARGS__)
#define PTX_DEBUG_MEM(fmt, ...)                                                \
    PTX_LOG_SIMPLE(ptxsim::log_level::debug, "mem", fmt, ##__VA_ARGS__)

#define PTX_TRACE_THREAD(fmt, ...)                                             \
    PTX_LOG_SIMPLE(ptxsim::log_level::trace, "thread", fmt, ##__VA_ARGS__)
#define PTX_DEBUG_THREAD(fmt, ...)                                             \
    PTX_LOG_SIMPLE(ptxsim::log_level::debug, "thread", fmt, ##__VA_ARGS__)

#define PTX_TRACE_EMU_SIMPLE(fmt, ...)                                         \
    PTX_LOG_SIMPLE(ptxsim::log_level::trace, "emu", fmt, ##__VA_ARGS__)
#define PTX_DEBUG_EMU_SIMPLE(fmt, ...)                                         \
    PTX_LOG_SIMPLE(ptxsim::log_level::debug, "emu", fmt, ##__VA_ARGS__)
#define PTX_INFO_EMU_SIMPLE(fmt, ...)                                          \
    PTX_LOG_SIMPLE(ptxsim::log_level::info, "emu_simple", fmt, ##__VA_ARGS__)
#endif

// 条件编译日志宏
#ifdef PTXSIM_DISABLE_LOGGING
#define PTX_TRACE_EMU_IF(cond, fmt, ...) ((void)0)
#define PTX_DEBUG_EMU_IF(cond, fmt, ...) ((void)0)
#else
#ifdef PTXSIM_USE_DETAILED_LOGGING
#define PTX_TRACE_EMU_IF(cond, fmt, ...)                                       \
    do {                                                                       \
        if ((cond) && ptxsim::LoggerConfig::get().is_enabled(                  \
                          ptxsim::log_level::trace, "emu")) {                  \
            auto loc = std::source_location::current();                        \
            std::string msg =                                                  \
                ptxsim::detail::printf_format(fmt, ##__VA_ARGS__);             \
            ptxsim::detail::output_log(ptxsim::log_level::trace, "emu", msg,   \
                                       loc);                                   \
        }                                                                      \
    } while (0)

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
#define PTX_TRACE_EMU_IF(cond, fmt, ...)                                       \
    do {                                                                       \
        if ((cond) && ptxsim::LoggerConfig::get().is_enabled(                  \
                          ptxsim::log_level::trace, "emu")) {                  \
            std::string msg =                                                  \
                ptxsim::detail::printf_format(fmt, ##__VA_ARGS__);             \
            ptxsim::detail::output_log_simple(ptxsim::log_level::trace, "emu", \
                                              msg);                            \
        }                                                                      \
    } while (0)

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

// 函数入口跟踪
#ifdef PTXSIM_USE_DETAILED_LOGGING
#define PTX_TRACE_FUNC()                                                       \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::trace,   \
                                                   "")) {                      \
            auto loc = std::source_location::current();                        \
            std::string short_name =                                           \
                ptxsim::detail::short_function_name(loc.function_name());      \
            ptxsim::detail::output_log(ptxsim::log_level::trace, "func",       \
                                       ptxsim::detail::printf_format(          \
                                           "[ENTER] %s", short_name.c_str()),  \
                                       loc);                                   \
        }                                                                      \
    } while (0)
#else
#define PTX_TRACE_FUNC()                                                       \
    do {                                                                       \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::trace,   \
                                                   "")) {                      \
            auto loc = std::source_location::current();                        \
            std::string short_name =                                           \
                ptxsim::detail::short_function_name(loc.function_name());      \
            ptxsim::detail::output_log_simple(                                 \
                ptxsim::log_level::trace, "func",                              \
                ptxsim::detail::printf_format("[ENTER] %s",                    \
                                              short_name.c_str()));            \
        }                                                                      \
    } while (0)
#endif

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