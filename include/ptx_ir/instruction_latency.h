#ifndef INSTRUCTION_LATENCY_H
#define INSTRUCTION_LATENCY_H

#include <cstdint>

namespace ptxsim {

struct InstructionLatency {
    uint32_t cycles;
    bool is_long_delay;
};

constexpr InstructionLatency DEFAULT_LATENCY{1, false};
constexpr InstructionLatency LD_GLOBAL_LATENCY{100, true};
constexpr InstructionLatency ST_GLOBAL_LATENCY{1, false};
constexpr InstructionLatency MUL_LATENCY{4, false};
constexpr InstructionLatency DIV_LATENCY{40, true};
constexpr InstructionLatency BAR_SYNC_LATENCY{1, false};

}
#endif