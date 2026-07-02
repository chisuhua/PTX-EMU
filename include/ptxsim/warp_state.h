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

    void reset() {
        for (auto &thread : threads) {
            thread.reset();
        }
        exec_mask = 0xFFFFFFFF;
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
