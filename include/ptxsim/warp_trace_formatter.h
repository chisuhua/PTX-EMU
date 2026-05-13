#ifndef WARP_TRACE_FORMATTER_H
#define WARP_TRACE_FORMATTER_H

#include "ptxsim/simt_stack.h"
#include "ptxsim/thread_state.h"
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ptxsim {

/**
 * @brief Warp 执行轨迹格式化器
 *
 * 将 warp 执行状态、SIMT 栈操作和线程分流信息格式化为紧凑的文本输出。
 * 所有方法均为静态，无状态开销。
 */
class WarpTraceFormatter {
public:
  /**
   * @brief 格式化单条指令执行记录
   *
   * @param cycle 当前周期计数
   * @param sm_id SM 标识
   * @param warp_id Warp 标识
   * @param pc 当前 PC
   * @param instruction_text 指令文本
   * @param active_mask 活跃线程掩码
   * @return 格式化后的字符串，例如：
   *         "Cycle 5: SM 0 Warp 0 PC=4  [全部32线程] @%p1 bra"
   */
  static std::string format_instruction(uint64_t cycle, int sm_id,
                                        int warp_id, int pc,
                                        const std::string& instruction_text,
                                        uint32_t active_mask);

  /**
   * @brief 格式化 SIMT 栈 push 事件
   *
   * @param cycle 当前周期计数
   * @param sm_id SM 标识
   * @param warp_id Warp 标识
   * @param entry 被推入的 SIMT 栈条目
   * @param taken_mask 执行分支的线程掩码
   * @return 格式化后的字符串
   */
  static std::string format_simt_push(uint64_t cycle, int sm_id, int warp_id,
                                      const SIMTStackEntry& entry,
                                      uint32_t taken_mask);

  /**
   * @brief 格式化 SIMT 栈 pop 事件
   *
   * @param cycle 当前周期计数
   * @param sm_id SM 标识
   * @param warp_id Warp 标识
   * @param popped_entry 被弹出的 SIMT 栈条目
   * @return 格式化后的字符串
   */
  static std::string format_simt_pop(uint64_t cycle, int sm_id, int warp_id,
                                     const SIMTStackEntry& popped_entry);

  /**
   * @brief 将线程掩码格式化为紧凑的范围表示
   *
   * @param mask 32 位线程掩码
   * @return 例如："全部32线程", "thread0", "thread0~15, thread17~31"
   */
  static std::string format_lane_ranges(uint32_t mask);

  /**
   * @brief 格式化线程分流信息（分歧路径）
   *
   * @param pc_to_lanes PC 到 lane 列表的映射
   * @return 格式化后的字符串，例如：
   *         "线程分流: thread0→PC=5, thread1~31→PC=8"
   */
  static std::string format_divergence(
      const std::map<int, std::vector<int>>& pc_to_lanes);

private:
  /**
   * @brief 将掩码转换为 "threadX~Y" 范围列表
   */
  static std::string mask_to_ranges(uint32_t mask);

  /**
   * @brief 将掩码格式化为十六进制字符串
   */
  static std::string mask_to_hex(uint32_t mask);
};

} // namespace ptxsim

#endif // WARP_TRACE_FORMATTER_H
