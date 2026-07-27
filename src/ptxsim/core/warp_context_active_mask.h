#ifndef PTXSIM_CORE_WARP_CONTEXT_ACTIVE_MASK_H
#define PTXSIM_CORE_WARP_CONTEXT_ACTIVE_MASK_H

#include <cstdint>

class WarpContext;  // forward decl (WarpContext lives in global namespace)

namespace warp_active_mask {

// Per-lane single-bit setter (called from divergence/convergence paths).
// Maintains active_count incrementally.
void set_active_mask_lane(WarpContext* w, int lane_id, bool active);

// 32-bit overwrite setter (called by ret handler to clear all lanes).
// CRITICAL: overwrite semantics, NOT OR-merge. ret handler at
// src/ptxsim/instructions/call.cpp:29 uses set_active_mask(0u) to clear
// all lanes after ret. OR-merge lives in BarrierModule::release_warp_barrier
// (T2-1 contract). See src/ptxsim/core/AGENTS.md for the full invariant set.
void set_active_mask_u32(WarpContext* w, uint32_t mask);

// Reads active_mask[] and returns a 32-bit packed bitmask.
uint32_t get_active_mask_u32(const WarpContext* w);

// Recomputes active_mask[] from warp_state.threads[i] (is_active/is_exited/
// is_blocked/status). Bidirectional: writes is_active back to warp_state
// to keep active_mask[] and warp_state.is_active synchronized.
// Invoked at the end of execute_warp_instruction().
void update_active_mask(WarpContext* w);

}  // namespace warp_active_mask
#endif