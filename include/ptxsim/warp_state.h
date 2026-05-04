#ifndef WARP_STATE_H
#define WARP_STATE_H

#include "ptxsim/thread_state.h"
#include "ptxsim/wbar.h"
#include <array>
#include <cstdint>
#include <map>
#include <string>

namespace ptxsim {

struct WarpState {
    std::array<ThreadState, 32> threads;
    uint32_t exec_mask = 0xFFFFFFFF;
    std::map<std::string, std::array<bool, 32>> thread_predicates;
    std::array<Wbar, 4> wbars;
    int current_wbar_id = -1;
    uint32_t warp_pc = 0;
    // pc_stack 和 pc_stack_depth 已移除 — 使用 WarpContext::pc_stacks 或 warp_state.threads[i].pc 替代

    void reset() {
        for (auto& thread : threads) {
            thread.reset();
        }
        exec_mask = 0xFFFFFFFF;
        thread_predicates.clear();
        for (auto& wbar : wbars) {
            wbar.reset();
        }
        current_wbar_id = -1;
        warp_pc = 0;
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
        for (const auto& thread : threads) {
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

}

#endif
