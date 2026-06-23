#ifndef PTXSIM_CONTEXTS_EXEC_STATE_H
#define PTXSIM_CONTEXTS_EXEC_STATE_H

#include "ptxsim/execution_types.h"
#include <cstdint>

namespace ptxsim {
namespace contexts {

/**
 * @brief Execution state POD: per-thread execution bookkeeping.
 *
 * @details Groups fields that describe WHERE this thread is in the execution
 *          hierarchy and WHAT state it is in. Pure data — no methods, no
 *          behavior. Encapsulates the per-thread identity (warp/lane/block
 *          indices) plus runtime state machine state (EXE_STATE, bar_id).
 *
 * @author PTX-EMU Team (T2-3 god-class split)
 * @date 2026-06-24
 */
struct ExecStatePod {
    // Thread identity within execution hierarchy
    Dim3 BlockIdx{0, 0, 0};
    Dim3 ThreadIdx{0, 0, 0};
    Dim3 GridDim{1, 1, 1};
    Dim3 BlockDim{1, 1, 1};
    int warp_id_ = 0;
    int lane_id_ = 0;

    // Runtime state machine
    EXE_STATE state = IDLE;
    int bar_id = 0;
};

}  // namespace contexts
}  // namespace ptxsim

#endif  // PTXSIM_CONTEXTS_EXEC_STATE_H