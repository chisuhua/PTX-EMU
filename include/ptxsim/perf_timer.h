#ifndef PERF_TIMER_H
#define PERF_TIMER_H

#include "utils/logger.h"
#include <chrono>

namespace ptxsim {

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

} // namespace ptxsim

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

#endif // PERF_TIMER_H