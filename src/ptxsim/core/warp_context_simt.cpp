#include "warp_context_simt.h"

#include "ptxsim/warp_context.h"
#include "ptxsim/simt_stack.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/ptx_config.h"
#include "ptxsim/warp_trace_formatter.h"
#include "ptxsim/sm_context.h"

namespace warp_simt {

bool check_reconvergence(WarpContext* w) {
    if (w->simt_stack.empty())
        return false;

    size_t depth_before = w->simt_stack.depth();

    // Check if converged; if so, capture the popped entry for tracing.
    ptxsim::SIMTStackEntry popped_entry;
    bool will_pop = w->simt_stack.top().is_converged(w->warp_state.threads);
    if (will_pop) {
        popped_entry = w->simt_stack.top();
    }

    w->simt_stack.check_reconvergence(w->warp_state.threads);

    if (w->simt_stack.depth() < depth_before) {
        int reconv_pc = popped_entry.reconvergence_pc;
        for (int i = 0; i < WarpContext::WARP_SIZE; i++) {
            if (!w->warp_state.threads[i].is_exited &&
                (int)w->warp_state.threads[i].pc == reconv_pc) {
                w->warp_state.threads[i].is_blocked = false;
                w->warp_state.threads[i].is_active = true;
            }
        }
        w->update_active_mask();
        if (w->simt_stack.empty()) {
            w->warp_state.exec_mask = 0xFFFFFFFF;
        } else {
            w->warp_state.exec_mask = w->simt_stack.top().return_mask;
        }
        // SIMT stack pop tracing
        if (ptxsim::DebugConfig::get().is_trace_simt_stack_enabled() &&
            w->sm_context_) {
            PTX_DEBUG_EMU("%s",
                          ptxsim::WarpTraceFormatter::format_simt_pop(
                              w->sm_context_->get_cycle_count(),
                              w->sm_context_->get_sm_id(), w->warp_id,
                              popped_entry)
                              .c_str());
        }
        return true;
    }
    return false;
}

}  // namespace warp_simt