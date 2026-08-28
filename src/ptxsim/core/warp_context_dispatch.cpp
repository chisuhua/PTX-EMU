#include "warp_context_dispatch.h"

#include "ptxsim/warp_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/thread_state.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/simt_stack.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/barrier/barrier_module.h"
#include "ptxsim/barrier/barrier_types.h"
#include "ptxsim/ptx_config.h"
#include "ptxsim/warp_trace_formatter.h"
#include "ptxsim/execution_trace.h"
#include "ptxsim/execution_types.h"
#include "utils/logger.h"
#include <cstdint>
#include <map>
#include <vector>

namespace warp_dispatch {

void execute_warp_instruction(WarpContext* w,
                               ptxemu::ir::StatementContext& stmt,
                               int target_pc) {
    std::vector<int> blocked_lanes;
    if (w->check_and_block_at_reconvergence_point(target_pc, blocked_lanes)) {
        // Capture lanes BEFORE update_active_mask (filtered otherwise).
        auto current_lanes_before_block = w->get_lanes_by_pc();
        w->update_active_mask();
        // Convergence-point debug trace.
        if (ptxsim::DebugConfig::get().is_trace_convergence_enabled() &&
            w->sm_context_) {
            auto current_lanes = current_lanes_before_block;
            if (current_lanes.size() > 1) {
                int next_pc = -1;
                uint32_t next_mask = 0;
                for (const auto& [candidate_pc, candidate_lanes] : current_lanes) {
                    if (candidate_pc == target_pc)
                        continue;
                    bool has_non_blocked = false;
                    for (int lane : candidate_lanes) {
                        if (!w->warp_state.threads[lane].is_blocked) {
                            has_non_blocked = true;
                            break;
                        }
                    }
                    if (has_non_blocked) {
                        next_pc = candidate_pc;
                        for (int lane : candidate_lanes) {
                            if (!w->warp_state.threads[lane].is_blocked)
                                next_mask |= (1u << lane);
                        }
                        break;
                    }
                }
                std::map<int, std::vector<int>> remaining_lanes;
                for (const auto& [pc_val, lane_list] : current_lanes) {
                    if (pc_val != target_pc) {
                        remaining_lanes[pc_val] = lane_list;
                    }
                }
                if (!remaining_lanes.empty() && next_pc >= 0) {
                    PTX_DEBUG_EMU("%s", ptxsim::WarpTraceFormatter::
                                            format_convergence_remaining(
                                                target_pc, remaining_lanes,
                                                next_pc, next_mask)
                                                .c_str());
                }
            }
        }
        return;
    }

    // Snapshot lanes to process BEFORE any handler runs.
    std::vector<int> lanes_to_process;
    for (int i = 0; i < WarpContext::WARP_SIZE; i++) {
        if (i >= (int)w->threads.size() || w->threads[i] == nullptr)
            continue;
        bool lane_active = w->is_lane_active(i);
        bool blocked_at_barrier = (w->threads[i]->get_state() == BAR_SYNC);
        if ((!lane_active && !blocked_at_barrier) ||
            w->warp_state.threads[i].pc != static_cast<uint32_t>(target_pc))
            continue;
        lanes_to_process.push_back(i);
    }

    for (int i : lanes_to_process) {
        ThreadContext* thread = w->threads[i].get();

        thread->sync_from_warp_state();

        // Re-check PC after sync: previous lane's divergent branch may
        // have moved this lane's PC away from target_pc.
        if (w->warp_state.threads[i].pc != static_cast<uint32_t>(target_pc)) {
            thread->sync_to_warp_state();
            continue;
        }

        // Skip already-exited lanes (warp-level handlers like ret mark ALL).
        if (thread->get_state() == EXIT) {
            thread->sync_to_warp_state();
            continue;
        }

        if (thread->get_state() == BAR_SYNC) {
            if (w->cta_context_ != nullptr) {
                auto& bm = w->cta_context_->get_barrier_module();
                bool any_wbar_incomplete = false;
                for (int j = 0; j < ptxsim::MAX_WARP_BARRIERS; ++j) {
                    auto* wbar = bm.get_warp_barrier(j);
                    if (wbar && wbar->is_initialized() && !wbar->is_complete()) {
                        any_wbar_incomplete = true;
                        break;
                    }
                }
                if (any_wbar_incomplete) {
                    PTX_WARN_EMU("Fallback CTA sync: lane %d, wbar incomplete",
                                 thread->lane_id_);
                    w->cta_context_->get_barrier_module().arrive_at_cta_barrier(
                        thread->bar_id, thread);
                }
            }
            thread->sync_to_warp_state();
            continue;
        }

        thread->execute_thread_instruction();
        thread->sync_to_warp_state();

        if (ptxsim::ExecutionTracer::is_enabled()) {
            ptxsim::ExecutionTracer::record(i, w->warp_state.threads[i].pc,
                                            stmt.instructionText);
        }
    }

    w->update_active_mask();
}

}  // namespace warp_dispatch