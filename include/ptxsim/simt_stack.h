#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <array>

#include "ptxsim/thread_state.h"

namespace ptxsim {

struct SIMTStackEntry {
    int branch_pc;
    int reconvergence_pc;
    uint32_t active_mask;
    uint32_t return_mask;
    int return_pc;
    
    bool is_converged(const std::array<ThreadState, 32>& threads) const;
    std::string toString() const;
};

class SIMTStack {
public:
    SIMTStack() = default;
    static constexpr size_t MAX_DEPTH = 32;

    void push(const SIMTStackEntry& entry);
    SIMTStackEntry pop();
    SIMTStackEntry& top();
    const SIMTStackEntry& top() const;
    const SIMTStackEntry& get_entry_at(size_t index) const;

    bool empty() const;
    size_t depth() const;
    void clear();

    bool check_reconvergence(const std::array<ThreadState, 32>& threads);
    void print() const;

private:
    std::vector<SIMTStackEntry> entries_;
};

}
