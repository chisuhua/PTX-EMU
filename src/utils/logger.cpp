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
        set_global_level_from_string(env_level);
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

    // 不在构造函数中加载配置文件，避免潜在的死锁
    // 配置文件将在其他地方加载
}

} // namespace ptxsim