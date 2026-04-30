#ifndef EXECUTION_TRACE_H
#define EXECUTION_TRACE_H

#include <array>
#include <string>
#include <vector>
#include <cstdint>

namespace ptxsim {

/** @brief 单条指令的执行记录 */
struct TraceEntry {
  uint32_t pc;                  // 当前 PC
  std::string instruction_text; // 完整指令文本

  bool operator==(const TraceEntry& other) const {
    return pc == other.pc && instruction_text == other.instruction_text;
  }
};

/** @brief 单个线程的完整执行轨迹 */
struct ThreadTrace {
  int lane_id;
  std::vector<TraceEntry> entries;
};

/** @brief Warp 级别（32 线程）的完整执行轨迹 */
struct ExecutionTrace {
  std::array<ThreadTrace, 32> threads;

  ExecutionTrace() {
    for (int i = 0; i < 32; i++) {
      threads[i].lane_id = i;
    }
  }

  void reset() {
    for (int i = 0; i < 32; i++) {
      threads[i].entries.clear();
    }
  }

  bool operator==(const ExecutionTrace& other) const {
    for (int i = 0; i < 32; i++) {
      const auto& t1 = threads[i].entries;
      const auto& t2 = other.threads[i].entries;
      if (t1.size() != t2.size()) return false;
      for (size_t j = 0; j < t1.size(); j++) {
        if (!(t1[j] == t2[j])) return false;
      }
    }
    return true;
  }
};

/**
 * @brief 全局执行跟踪器
 *
 * 在 warp 执行循环中，每条指令执行后调用 record() 记录轨迹。
 * 通过 is_enabled() 控制是否启用，零开销关闭。
 */
class ExecutionTracer {
public:
  static void enable() { enabled_ = true; }
  static void disable() { enabled_ = false; }
  static bool is_enabled() { return enabled_; }

  static void reset() { trace_.reset(); }

  static void record(int lane_id, uint32_t pc, const std::string& instruction) {
    if (!enabled_) return;
    if (lane_id < 0 || lane_id >= 32) return;
    trace_.threads[lane_id].entries.push_back({pc, instruction});
  }

  static const ExecutionTrace& get_trace() { return trace_; }

  static ExecutionTrace trace_copy() { return trace_; }

private:
  static bool enabled_;
  static ExecutionTrace trace_;
};

} // namespace ptxsim

#endif // EXECUTION_TRACE_H