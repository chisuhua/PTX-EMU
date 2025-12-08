// utils/logger.h
#ifndef PTX_LOGGER_H
#define PTX_LOGGER_H

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
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
#include <cstring>

namespace ptxsim {

// ===========================================================================
// 日志级别定义
// ===========================================================================

enum class log_level {
    trace = 0, // 比debug更详细，用于指令级跟踪
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
inline const char *reset_color() {
    return "\033[0m";
}

// 输出日志的核心函数声明
void output_log(log_level level, const std::string &component,
                const std::string &msg, const std::source_location &loc);

// 输出简单日志的核心函数声明
void output_log_simple(log_level level, const std::string &component,
                       const std::string &msg);

// 格式化函数模板
template <typename... Args>
std::string printf_format(const char *fmt, Args &&...args) {
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
            std::ifstream config_file(filename);
            if (!config_file.is_open()) {
                return false;
            }

            std::string line;
            while (std::getline(config_file, line)) {
                // 跳过注释和空行
                if (line.empty() || line[0] == '#')
                    continue;

                // 解析 key=value 格式的配置行
                size_t eq_pos = line.find('=');
                if (eq_pos == std::string::npos)
                    continue;

                std::string key = line.substr(0, eq_pos);
                std::string value = line.substr(eq_pos + 1);

                // 去除首尾空白字符
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                if (key == "global_level") {
                    set_global_level_from_string(value);
                } else if (key == "target") {
                    set_target_from_string(value);
                } else if (key == "logfile") {
                    set_logfile_internal(value);
                } else if (key == "async") {
                    use_async_logging_ = (value == "true" || value == "1");
                } else if (key == "colorize") {
                    format_options_.colorize =
                        (value == "true" || value == "1");
                } else if (key.find("component.") == 0) {
                    // 组件级别的配置
                    std::string component = key.substr(10); // 去掉 "component." 前缀
                    set_component_level_from_string(component, value);
                }
            }
            return true;
        } catch (...) {
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

    // 获取当前日志目标
    log_target get_target() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return target_;
    }

    // 禁用所有日志
    void disable_all() {
        set_global_level(log_level::fatal);
    }

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
    bool use_async_logging_;
    LogFormatOptions format_options_;
    std::unique_ptr<AsyncLogQueue> async_queue_;

    // 禁止拷贝构造和赋值
    LoggerConfig(const LoggerConfig &) = delete;
    LoggerConfig &operator=(const LoggerConfig &) = delete;
    
    // 友元声明，允许detail命名空间中的函数访问私有成员
    friend void detail::output_log(log_level level, const std::string &component,
                                  const std::string &msg,
                                  const std::source_location &loc);
    friend void detail::output_log_simple(log_level level,
                                         const std::string &component,
                                         const std::string &msg);
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

// 常用组件日志宏
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

// 条件编译日志宏
#ifdef PTXSIM_DISABLE_LOGGING
#define PTX_TRACE_EMU_IF(cond, fmt, ...) ((void)0)
#define PTX_DEBUG_EMU_IF(cond, fmt, ...) ((void)0)
#else
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
#endif

// 简洁版本（不包含位置信息，性能更好）
#define PTX_TRACE_EMU_SIMPLE(fmt, ...)                                         \
    PTX_LOG_SIMPLE(ptxsim::log_level::trace, "emu", fmt, ##__VA_ARGS__)
#define PTX_DEBUG_EMU_SIMPLE(fmt, ...)                                         \
    PTX_LOG_SIMPLE(ptxsim::log_level::debug, "emu", fmt, ##__VA_ARGS__)
#define PTX_INFO_EMU_SIMPLE(fmt, ...)                                          \
    PTX_LOG_SIMPLE(ptxsim::log_level::info, "emu_simple", fmt, ##__VA_ARGS__)

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
#define PTX_FATAL(component, fmt, ...)                                         \
    do {                                                                       \
        auto loc = std::source_location::current();                            \
        std::string msg =                                                      \
            ptxsim::detail::printf_format("FATAL: " fmt, ##__VA_ARGS__);       \
        ptxsim::detail::output_log(ptxsim::log_level::fatal, component, msg,   \
                                   loc);                                       \
    } while (0)

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