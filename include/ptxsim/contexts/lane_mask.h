#ifndef PTXSIM_CONTEXTS_LANE_MASK_H
#define PTXSIM_CONTEXTS_LANE_MASK_H

#include <array>
#include <cstdint>

namespace ptxsim {
namespace contexts {

/**
 * @brief Lane mask POD: per-warp lane-activity tracking.
 *
 * @details Groups the lane-activity tracking state: the bool[32] active_mask
 *          cache (recomputed by WarpContext::update_active_mask), the
 *          active_count counter, the warp_thread_ids array (logical
 *          thread index per lane), the divergence_detected flag, and
 *          the is_scheduled_ scheduler flag. Pure data — no methods.
 *
 * @author PTX-EMU Team (T2-3 god-class split)
 * @date 2026-06-24
 */
struct LaneMaskPod {
    static constexpr int WARP_SIZE = 32;

    // Per-lane active flag (derived cache; authoritative source is
    // warp_state.threads[i].is_schedulable() per T2-1)
    std::array<bool, WARP_SIZE> active_mask{};

    // Logical thread IDs per lane (warp_thread_ids[lane] = global thread id)
    std::array<int, WARP_SIZE> warp_thread_ids{};

    // Active-lane count (derived; recomputed by update_active_mask)
    int active_count = 0;

    // Divergence detected flag (set by handle_branch)
    bool divergence_detected = false;

    // Scheduler flag: warp is currently selected for execution
    bool is_scheduled_ = false;
};

}  // namespace contexts
}  // namespace ptxsim

#endif  // PTXSIM_CONTEXTS_LANE_MASK_H