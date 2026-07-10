#ifndef WARP_STATE_H
#define WARP_STATE_H

#include "ptxsim/thread_state.h"
#include <array>
#include <cstdint>

namespace ptxsim {

struct WarpState {
    // T2-3 A2: Reduced from 6 fields to 4 fields (threads[] + exec_mask + 2
    // deprecated). thread_predicates (0 production refs) and warp_pc
    // (0 production refs) physically removed.
    std::array<ThreadState, 32> threads;
    uint32_t exec_mask = 0xFFFFFFFF;

    // Phase 1 of implement-tcgen05-handlers-extended: per-warp
    // tcgen05.alloc permit. Defaults to true (warp may allocate). Set
    // to false by `tcgen05.relinquish_alloc_permit`; only set back to
    // true by CTAContext teardown (per PTX ISA §9.7.16: a warp that
    // relinquishes its permit must wait for CTA exit to re-acquire).
    bool allocate_permit = true;

    // Phase 4 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q6-B):
    // tcgen05.fence position marker (no-op semantics). Stores the last
    // fence type encountered by this warp (FencePosition enum at warp_context.h).
    // Defaults to kFenceNone (0). Reset on WarpState::reset() (CTA teardown).
    // Per ptx-lessons-learned §2: no mutex — single-writer (scheduler).
    int8_t fence_position = 0;

    void reset() {
        for (auto &thread : threads) {
            thread.reset();
        }
        exec_mask = 0xFFFFFFFF;
        allocate_permit = true;
        fence_position = 0;
    }

    int count_active_lanes() const {
        int count = 0;
        for (int i = 0; i < 32; ++i) {
            if (threads[i].is_active && !threads[i].is_exited) {
                ++count;
            }
        }
        return count;
    }

    int count_schedulable_lanes() const {
        int count = 0;
        for (int i = 0; i < 32; ++i) {
            if (threads[i].is_schedulable()) {
                ++count;
            }
        }
        return count;
    }

    bool is_all_exited() const {
        for (const auto &thread : threads) {
            if (!thread.is_exited) {
                return false;
            }
        }
        return true;
    }

    bool has_schedulable_threads() const {
        return count_schedulable_lanes() > 0;
    }
};

} // namespace ptxsim

#endif
