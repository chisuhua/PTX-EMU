#include "warp_context_active_mask.h"

#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/thread_state.h"

namespace warp_active_mask {

void set_active_mask_lane(WarpContext* w, int lane_id, bool active) {
    if (lane_id >= 0 && lane_id < WarpContext::WARP_SIZE) {
        bool was_active = w->active_mask[lane_id];
        w->active_mask[lane_id] = active;
        w->warp_state.threads[lane_id].is_active = active;

        if (was_active && !active) {
            w->active_count--;
        } else if (!was_active && active) {
            w->active_count++;
        }
    }
}

void update_active_mask(WarpContext* w) {
    w->active_count = 0;
    for (int i = 0; i < WarpContext::WARP_SIZE; i++) {
        if (i < w->threads.size() && w->threads[i] != nullptr) {
            bool active =
                w->warp_state.threads[i].is_active &&
                !w->warp_state.threads[i].is_exited &&
                !w->warp_state.threads[i].is_blocked &&
                (w->warp_state.threads[i].status == ptxsim::ThreadStatus::Active);
            w->active_mask[i] = active;
            w->warp_state.threads[i].is_active = active;
            if (active)
                w->active_count++;
        }
    }
}

uint32_t get_active_mask_u32(const WarpContext* w) {
    uint32_t mask = 0;
    for (int i = 0; i < WarpContext::WARP_SIZE && i < 32; i++) {
        if (w->active_mask[i]) {
            mask |= (1U << i);
        }
    }
    return mask;
}

void set_active_mask_u32(WarpContext* w, uint32_t mask) {
    // Overwrite semantics: ret handler uses set_active_mask(0u) to clear all
    // lanes after ret. Do NOT change to OR-merge — the OR pattern lives in
    // BarrierModule::release_warp_barrier (T2-1 contract).
    w->active_count = 0;
    for (int i = 0; i < WarpContext::WARP_SIZE && i < 32; i++) {
        bool active = (mask >> i) & 1;
        w->active_mask[i] = active;
        w->warp_state.threads[i].is_active = active;
        if (active) {
            w->active_count++;
        }
    }
}

}  // namespace warp_active_mask