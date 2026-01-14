#include "utils/logger.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace ptxsim {

LoggerConfig::LoggerConfig()
    : global_level_(log_level::info), target_(log_target::console),
      use_async_logging_(false) {
    // 从环境变量读取初始配置
    char *env_level = std::getenv("PTXSIM_LOG_LEVEL");
    if (env_level) {
        set_global_level(string_to_log_level(env_level));
    }

    char *env_target = std::getenv("PTXSIM_LOG_TARGET");
    if (env_target) {
        set_target_from_string(env_target);
    }

    char *env_async = std::getenv("PTXSIM_ASYNC_LOG");
    if (env_async) {
        use_async_logging_ = (std::strcmp(env_async, "1") == 0 ||
                              std::strcmp(env_async, "true") == 0);
    }

    format_options_.colorize = true;

    // 不在构造函数中加载配置文件，避免潜在的死锁
    // 配置文件将在其他地方加载
}

void LoggerConfig::load_from_ini_section(
    const inipp::Ini<char>::Section &logger_section) {
    // std::lock_guard<std::mutex> lock(mutex_);

    std::string level_str;
    inipp::get_value(logger_section, "global_level", level_str);
    if (!level_str.empty()) {
        set_global_level(string_to_log_level(level_str));
    }

    std::string target_str;
    inipp::get_value(logger_section, "target", target_str);
    if (!target_str.empty()) {
        set_target_from_string(target_str);
    }

    std::string logfile;
    inipp::get_value(logger_section, "logfile", logfile);
    if (!logfile.empty()) {
        logfile_path_ = logfile;
        // 只有当路径设置成功时才尝试打开文件
        set_logfile_internal(logfile);
    }

    std::string async_str;
    inipp::get_value(logger_section, "async", async_str);
    if (!async_str.empty()) {
        bool async = (async_str == "true" || async_str == "1");
        enable_async_logging(async);
    }

    std::string colorize_str;
    inipp::get_value(logger_section, "colorize", colorize_str);
    if (!colorize_str.empty()) {
        bool colorize = (colorize_str == "true" || colorize_str == "1");
        set_use_color_output(colorize);
    }


    // 读取组件级别配置
    for (const auto &pair : logger_section) {
        if (pair.first.length() > 10 &&
            pair.first.substr(0, 9) == "component") {
            std::string component = pair.first.substr(10); // skip "component."
            if (!component.empty() &&
                component[0] == '.') { // 确保格式为 "component.name"
                component = component.substr(1); // remove leading dot
                if (!component.empty()) {
                    log_level level = string_to_log_level(pair.second);
                    set_component_level(component, level);
                }
            }
        }
    }
}

} // namespace ptxsim