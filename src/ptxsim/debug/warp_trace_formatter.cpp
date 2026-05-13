#include "ptxsim/warp_trace_formatter.h"
#include <iomanip>
#include <sstream>

namespace ptxsim {

std::string WarpTraceFormatter::format_instruction(uint64_t cycle, int sm_id,
                                                   int warp_id, int pc,
                                                   const std::string& instruction_text,
                                                   uint32_t active_mask) {
  std::ostringstream oss;
  oss << "Cycle " << cycle << ": SM " << sm_id << " Warp " << warp_id
      << " PC=" << pc << "  [" << format_lane_ranges(active_mask) << "] "
      << instruction_text;
  return oss.str();
}

std::string WarpTraceFormatter::format_simt_push(uint64_t cycle, int sm_id,
                                                 int warp_id,
                                                 const SIMTStackEntry& entry,
                                                 uint32_t taken_mask) {
  std::ostringstream oss;
  oss << "         → SIMT Stack push: branch_pc=" << entry.branch_pc
      << ", reconvergence_pc=" << entry.reconvergence_pc << "\n"
      << "           taken_mask=0x" << std::hex << std::uppercase << taken_mask
      << std::dec << " (" << mask_to_ranges(taken_mask) << ")";
  return oss.str();
}

std::string WarpTraceFormatter::format_simt_pop(uint64_t cycle, int sm_id,
                                                int warp_id,
                                                const SIMTStackEntry& popped_entry) {
  std::ostringstream oss;
  oss << "         → SIMT Stack pop: reconvergence_pc="
      << popped_entry.reconvergence_pc;
  return oss.str();
}

std::string WarpTraceFormatter::format_lane_ranges(uint32_t mask) {
  if (mask == 0xFFFFFFFFu) {
    return "全部32线程";
  }
  if (mask == 0) {
    return "无活跃线程";
  }
  return mask_to_ranges(mask);
}

std::string WarpTraceFormatter::format_divergence(
    const std::map<int, std::vector<int>>& pc_to_lanes) {
  if (pc_to_lanes.size() <= 1) {
    return "";
  }

  std::ostringstream oss;
  oss << "线程分流: ";
  bool first = true;
  for (const auto& [pc, lanes] : pc_to_lanes) {
    if (!first) {
      oss << ", ";
    }
    first = false;

    // 将 lane 列表转换为掩码，再格式化
    uint32_t mask = 0;
    for (int lane : lanes) {
      if (lane >= 0 && lane < 32) {
        mask |= (1u << lane);
      }
    }
    oss << mask_to_ranges(mask) << "→PC=" << pc;
  }
  return oss.str();
}

std::string WarpTraceFormatter::mask_to_ranges(uint32_t mask) {
  if (mask == 0) {
    return "";
  }

  std::ostringstream oss;
  bool first = true;
  int start = -1;

  for (int i = 0; i <= 32; ++i) {
    bool bit_set = (i < 32) && (mask & (1u << i));
    if (bit_set && start == -1) {
      start = i;
    }
    if ((!bit_set || i == 32) && start != -1) {
      int end = i - 1;
      if (!first) {
        oss << ", ";
      }
      first = false;

      if (start == end) {
        oss << "thread" << start;
      } else {
        oss << "thread" << start << "~" << end;
      }
      start = -1;
    }
  }

  return oss.str();
}

std::string WarpTraceFormatter::mask_to_hex(uint32_t mask) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::uppercase << std::setfill('0')
      << std::setw(8) << mask;
  return oss.str();
}

} // namespace ptxsim
