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
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <vector>

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
// 终端颜色代码
inline const char *level_color(log_level level) {
    switch (level) {
    case log_level::trace:
        return "\033[37m"; // 灰色
    case log_level::debug:
        return "\033[36m"; // 青色
    case log_level::info:
        return "\033[32m"; // 绿色
    case log_level::warning:
        return "\033[33m"; // 黄色
    case log_level::error:
        return "\033[31m"; // 红色
    case log_level::fatal:
        return "\033[41;37m"; // 红底白字
    default:
        return "\033[0m";
    }
}

inline const char *reset_color() { return "\033[0m"; }

// printf 风格的格式化函数
template <typename... Args>
std::string printf_format(const char *fmt, Args &&...args) {
    // 预估缓冲区大小
    size_t size = std::snprintf(nullptr, 0, fmt, args...) + 1;
    if (size <= 0) {
        return std::string(fmt); // 格式化失败，返回原始格式字符串
    }

    // 分配缓冲区并格式化
    auto buf = std::make_unique<char[]>(size);
    std::snprintf(buf.get(), size, fmt, args...);
    return std::string(buf.get());
}

// 通用的变量到字符串转换
template <typename T> std::string to_string(const T &value) {
    if constexpr (std::is_arithmetic_v<T>) {
        if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
            return printf_format("%lld", static_cast<long long>(value));
        } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
            return printf_format("%llu",
                                 static_cast<unsigned long long>(value));
        } else if constexpr (std::is_floating_point_v<T>) {
            return printf_format("%g", static_cast<double>(value));
        } else {
            return printf_format("%lld", static_cast<long long>(value));
        }
    } else {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }
}

// 提取简短的函数名
inline std::string short_function_name(const char *full_name) {
    std::string name(full_name);
    // 查找最后一个 '::' 或者 '('
    size_t pos = name.rfind("::");
    if (pos != std::string::npos) {
        pos += 2; // 跳过 "::"
    } else {
        pos = 0;
    }
    size_t end_pos = name.find('(', pos);
    if (end_pos != std::string::npos) {
        return name.substr(pos, end_pos - pos);
    }
    return name.substr(pos);
}

// Flag to indicate if we're in static destruction phase
inline bool &in_static_destruction() {
    static bool flag = false;
    return flag;
}

// Function to set the static destruction flag
inline void set_static_destruction() { in_static_destruction() = true; }

// 获取当前时间戳（格式化为字符串）
inline std::string current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
    return ss.str();
}

// 获取当前线程ID（简化版）
inline std::string current_thread_id() {
    static thread_local std::string tid = std::to_string(
        std::hash<std::thread::id>{}(std::this_thread::get_id()));
    return tid;
}

// 获取当前进程ID
inline pid_t current_process_id() { return getpid(); }
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
    static LoggerConfig &get() {
        static LoggerConfig instance;
        return instance;
    }

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
                if (key == "global_level") {
                    set_global_level_from_string(value);
                } else if (key == "log_target") {
                    set_target_from_string(value);
                } else if (key == "log_file") {
                    set_logfile(value);
                } else if (key.find("component_level.") == 0) {
                    std::string component = key.substr(17);
                    set_component_level_from_string(component, value);
                } else if (key == "async_logging") {
                    enable_async_logging(value == "true" || value == "1");
                } else if (key == "colorize") {
                    format_options_.colorize =
                        (value == "true" || value == "1");
                }
            }
            return true;
        } catch (...) {
            return false;
        }
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

    // 设置日志文件
    bool set_logfile(const std::string &filename) {
        std::lock_guard<std::mutex> lock(mutex_);
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
        if (detail::in_static_destruction())
            return false;

        std::lock_guard<std::mutex> lock(mutex_);
        log_level effective_level =
            component.empty() ? global_level_ : get_effective_level(component);

        return static_cast<int>(level) >= static_cast<int>(effective_level);
    }

    // 获取当前日志目标
    log_target get_target() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return target_;
    }

    // 禁用所有日志
    void disable_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        global_level_ = log_level::fatal;
        component_levels_.clear();
    }

    // 启用/禁用异步日志记录
    void enable_async_logging(bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);
        use_async_logging_ = enable;
    }

    // 获取日志格式选项
    const LogFormatOptions &get_format_options() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return format_options_;
    }

    // 设置日志格式选项
    void set_format_options(const LogFormatOptions &options) {
        std::lock_guard<std::mutex> lock(mutex_);
        format_options_ = options;
    }

    ~LoggerConfig() {
        detail::set_static_destruction();
        if (logfile_.is_open()) {
            logfile_.close();
        }
    }

private:
    LoggerConfig()
        : global_level_(log_level::info), target_(log_target::console),
          use_async_logging_(false) {
        // 检查环境变量
        char *env_level = std::getenv("PTXSIM_LOG_LEVEL");
        if (env_level) {
            set_global_level_from_string(env_level);
        }

        char *env_file = std::getenv("PTXSIM_LOG_FILE");
        if (env_file) {
            if (set_logfile(env_file)) {
                target_ = log_target::both;
            }
        }

        char *env_async = std::getenv("PTXSIM_ASYNC_LOG");
        if (env_async) {
            use_async_logging_ = (std::strcmp(env_async, "1") == 0 ||
                                  std::strcmp(env_async, "true") == 0);
        }

        char *env_config = std::getenv("PTXSIM_LOG_CONFIG");
        if (env_config) {
            load_from_file(env_config);
        }
    }

    void set_global_level_from_string(const std::string &level_str) {
        if (level_str == "trace")
            global_level_ = log_level::trace;
        else if (level_str == "debug")
            global_level_ = log_level::debug;
        else if (level_str == "info")
            global_level_ = log_level::info;
        else if (level_str == "warning")
            global_level_ = log_level::warning;
        else if (level_str == "error")
            global_level_ = log_level::error;
        else if (level_str == "fatal")
            global_level_ = log_level::fatal;
    }

    void set_component_level_from_string(const std::string &component,
                                         const std::string &level_str) {
        log_level level = log_level::info;
        if (level_str == "trace")
            level = log_level::trace;
        else if (level_str == "debug")
            level = log_level::debug;
        else if (level_str == "warning")
            level = log_level::warning;
        else if (level_str == "error")
            level = log_level::error;
        else if (level_str == "fatal")
            level = log_level::fatal;

        component_levels_[component] = level;
    }

    void set_target_from_string(const std::string &target_str) {
        if (target_str == "console")
            target_ = log_target::console;
        else if (target_str == "file")
            target_ = log_target::file;
        else if (target_str == "both")
            target_ = log_target::both;
    }

    mutable std::mutex mutex_;
    log_level global_level_;
    std::unordered_map<std::string, log_level> component_levels_;
    log_target target_;
    std::ofstream logfile_;
    bool use_async_logging_;
    LogFormatOptions format_options_;
    std::unique_ptr<AsyncLogQueue> async_queue_;

    LoggerConfig(const LoggerConfig &) = delete;
    LoggerConfig &operator=(const LoggerConfig &) = delete;
};

// ===========================================================================
// 基础日志输出函数
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

// 线程安全的日志输出
inline void output_log(log_level level, const std::string &component,
                       const std::string &message,
                       const std::source_location &loc) {
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
        if (func_ && !detail::in_static_destruction()) {
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

} // namespace ptxsim

#endif // PTX_LOGGER_H