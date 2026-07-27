#include "sm_context_reconvergence.h"

#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/simt_stack.h"
#include "ptxsim/ptx_config.h"
#include "ptxsim/warp_trace_formatter.h"
#include "utils/logger.h"

namespace sm_reconvergence {

// Drain the SIMT stack until no more convergent entries can be popped.
// Extracted from sm_context.cpp:455-490 and :580-623 (two near-duplicate
// blocks). Preserves the convergence trace output verbatim.
//
// After the loop, calls update_active_mask() so downstream gating (such
// as the schedulable lane check in the next iteration of the SM cycle)
// sees the post-reconvergence active set without waiting for the next
// execute_warp_instruction() to recompute it (lessons-learned §1 contract).
void drain_simt_and_update_active(WarpContext* warp) {
    size_t stack_depth_before = warp->get_simt_stack().depth();
    int reconvergence_pc = -1;
    if (!warp->get_simt_stack().empty()) {
        reconvergence_pc = warp->get_simt_stack().top().reconvergence_pc;
    }
    while (warp->check_reconvergence()) {
        // Keep popping until no more convergent entries
    }
    // 汇聚点调试输出：检测到 SIMT 栈弹出（reconvergence）
    if (ptxsim::DebugConfig::get().is_trace_convergence_enabled() &&
        warp->get_simt_stack().depth() < stack_depth_before) {
        auto current_lanes = warp->get_lanes_by_pc();
        if (current_lanes.size() == 1) {
            uint32_t merged_mask = 0;
            for (int lane : current_lanes.begin()->second) {
                merged_mask |= (1u << lane);
            }
            int merged_pc = current_lanes.begin()->first;
            PTX_DEBUG_EMU("%s", ptxsim::WarpTraceFormatter::
                                    format_reconvergence(
                                        reconvergence_pc,
                                        merged_pc, merged_mask)
                                    .c_str());
        }
    }
    warp->update_active_mask();
}

}  // namespace sm_reconvergence