#ifndef WARP_STATE_H
#define WARP_STATE_H

#include "ptxsim/thread_state.h"
#include "ptxsim/wbar.h"
#include <array>
#include <cstdint>

namespace ptxsim {

struct WarpState {
    // T2-3 A2: Reduced from 6 fields to 4 fields (threads[] + exec_mask + 2
    // deprecated). thread_predicates (0 production refs) and warp_pc
    // (0 production refs) physically removed.
    std::array<ThreadState, 32> threads;
    uint32_t exec_mask = 0xFFFFFFFF;

    // T2-1 Task 5 + T2-3 A2/A5: Deprecated. Production barrier handlers
    // (BarHandler, BarWarpSyncHandler at src/ptxsim/instructions/barrier.cpp
    // still route through this legacy API). integrate-barrier-module-cta-warp
    // is the blocker change. Once that change is merged, T2-3 A5 will
    // physically remove wbars[] + current_wbar_id.
    [[deprecated("Use BarrierModule::get_warp_barrier() instead — will be "
                 "removed in T2-3 A5 after integrate-barrier-module-cta-warp "
                 "merges")]]
    std::array<Wbar, 4> wbars;
    [[deprecated("Use BarrierModule state instead — will be removed in T2-3 A5 "
                 "after integrate-barrier-module-cta-warp merges")]]
    int current_wbar_id = -1;
    // pc_stack 和 pc_stack_depth 已移除 — 使用 WarpContext::pc_stacks 或
    // warp_state.threads[i].pc 替代

    void reset() {
        for (auto &thread : threads) {
            thread.reset();
        }
        exec_mask = 0xFFFFFFFF;
        for (auto &wbar : wbars) {
            wbar.reset();
        }
        current_wbar_id = -1;
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
