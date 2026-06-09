#include "ptx_ir/instruction_latency_table.h"

namespace ptxsim {

// Dispatcher for instruction latency. StatementType values come from
// ptx_op.def; cases below override DEFAULT_LATENCY for known
// long-latency / multi-cycle instructions.
InstructionLatency getLatency(StatementType type) {
    switch (type) {
    case S_LD:
        return LD_GLOBAL_LATENCY;
    case S_ST:
        return ST_GLOBAL_LATENCY;
    case S_MUL:
    case S_MUL24:
    case S_MAD:
    case S_MAD24:
    case S_FMA:
        return MUL_LATENCY;
    case S_DIV:
    case S_REM:
        return DIV_LATENCY;
    case S_BAR:
    case S_BAR_WARP_SYNC:
    case S_MEMBAR:
    case S_FENCE:
        return BAR_SYNC_LATENCY;
    default:
        return DEFAULT_LATENCY;
    }
}

} // namespace ptxsim
