/**
 * @file debug_example.cpp
 * @brief PTX-EMU 调试功能使用示例
 *
 * 这个示例程序展示了如何在 PTX-EMU 中使用调试功能，
 * 包括加载配置文件、设置断点和查看寄存器状态等。
 */

#include "ptxsim/interpreter.h"
#include "ptxsim/ptx_debug.h"
#include "ptxsim/thread_context.h"
#include "utils/logger.h"

#include <iostream>
#include <memory>

using namespace ptxsim;

/**
 * @brief 初始化调试环境
 */
void initialize_debug_environment(const std::string &config_path = "") {
    // 获取日志配置实例
    auto &logger_config = LoggerConfig::get();

    // 如果提供了配置文件路径，则尝试加载它
    if (!config_path.empty()) {
        bool loaded = logger_config.load_from_file(config_path);
        if (!loaded) {
            std::cout << "Warning: Failed to load debug configuration file: "
                      << config_path << ". Using default settings.\n";
            // 设置默认调试级别
            logger_config.set_global_level(log_level::debug);
        } else {
            std::cout << "Debug configuration loaded from " << config_path
                      << "\n";
        }
    } else {
        // 尝试加载默认配置文件
        bool loaded = logger_config.load_from_file("ptx_debug.conf");
        if (!loaded) {
            std::cout << "Warning: Failed to load debug configuration file. "
                         "Using default settings.\n";
            // 设置默认调试级别
            logger_config.set_global_level(log_level::debug);
        } else {
            std::cout << "Debug configuration loaded from default file\n";
        }
    }

    std::cout << "Debug environment initialized.\n";
}

/**
 * @brief 演示线程上下文调试功能
 */
void demonstrate_thread_context_debug() {
    std::cout << "\n=== Thread Context Debug Demo ===\n";

    // 创建一个简单的线程上下文示例
    // 注意：在真实环境中，这些会由解释器自动创建和管理
    /*
    ThreadContext context(0, 0, 0); // cta_id, warp_id, thread_id

    // 模拟设置一些寄存器值
    context.setRegister("r0", 10);
    context.setRegister("r1", 20);
    context.setRegister("r2", 30);

    // 演示 dump_state 功能
    std::cout << "Dumping thread state:\n";
    context.dump_state();

    // 演示 prepare_breakpoint_context 功能
    std::cout << "\nPreparing breakpoint context:\n";
    std::map<std::string, uint64_t> bp_context;
    context.prepare_breakpoint_context(bp_context);

    std::cout << "Breakpoint context prepared with " << bp_context.size() << "
    entries.\n"; for (const auto& entry : bp_context) { std::cout << "  " <<
    entry.first << " = " << entry.second << "\n";
    }
    */
}

/**
 * @brief 演示性能计时器功能
 */
void demonstrate_perf_timer() {
    std::cout << "\n=== Performance Timer Demo ===\n";

    // PerfTimer 会在析构时自动输出性能统计信息
    // 这里只是演示它的存在，实际使用时会在指令执行等地方自动使用
    std::cout << "PerfTimer can be used to measure execution performance.\n";
    std::cout
        << "It automatically outputs statistics when going out of scope.\n";
}

/**
 * @brief 主函数
 */
int main(int argc, char *argv[]) {
    std::cout << "PTX-EMU Debug Features Demonstration\n";
    std::cout << "=====================================\n";

    // 检查是否有配置文件参数
    std::string config_path = "";
    if (argc > 1) {
        config_path = argv[1];
    }

    // 初始化调试环境
    initialize_debug_environment(config_path);

    // 演示各种调试功能
    demonstrate_thread_context_debug();
    demonstrate_perf_timer();

    // 示例日志输出
    PTX_DEBUG_EMU("This is a debug message from the emulator");
    PTX_INFO_EMU("This is an info message from the emulator");
    // PTX_TRACE_EMU("This is a trace message from the emulator");

    std::cout << "\nDemo completed. Check the log output and log file (if "
                 "configured).\n";

    return 0;
}