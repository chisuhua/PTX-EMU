// assertion_utils.h
#ifndef PTXSIM_TESTING_ASSERTION_UTILS_H
#define PTXSIM_TESTING_ASSERTION_UTILS_H

#include "ptxsim/warp_context.h"

#include <cstdint>
#include <iostream>
#include <sstream>

namespace ptxsim::testing {

// ============================================================================
// Verification Helpers
// Count and check warp state for PTX simulation testing
// ============================================================================

// Count number of active lanes in the warp
inline int count_active_lanes(const WarpContext& warp) {
    int n = 0;
    for (int i = 0; i < 32; i++)
        if (warp.is_lane_active(i)) n++;
    return n;
}

// Count number of threads at a specific PC
inline int count_at_pc(const WarpContext& warp, uint32_t pc) {
    int n = 0;
    for (int i = 0; i < 32; i++)
        if (warp.get_warp_state().threads[i].pc == pc) n++;
    return n;
}

// Get the current active mask from warp
inline uint32_t get_active_mask(const WarpContext& warp) {
    return warp.get_active_mask();
}

// Check that the warp's active mask matches expected value
// Returns true if mask matches, false otherwise
// Prints details to os stream
inline bool check_mask(WarpContext& warp, uint32_t expected, std::ostream& os = std::cerr) {
    uint32_t actual = warp.get_active_mask();
    bool matches = (actual == expected);
    os << "mask: expected=0x" << std::hex << expected << ", got=0x" << actual;
    if (!matches) {
        os << " [MISMATCH]";
    }
    os << std::dec << std::endl;
    return matches;
}

} // namespace ptxsim::testing

#endif // PTXSIM_TESTING_ASSERTION_UTILS_H