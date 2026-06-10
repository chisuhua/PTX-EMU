#ifndef INSTRUCTION_LATENCY_CONFIG_H
#define INSTRUCTION_LATENCY_CONFIG_H

#include <cstdint>

// Per-instruction-class latency overrides loaded from the GPU JSON
// "instruction_latencies" block. cycles <= 0 falls back to the
// constexpr defaults in ptx_ir/instruction_latency.h. Values are
// interpreted as "core cycles" (NOT nanoseconds) so the existing
// sm_context.cpp blocked-cycles decrement loop keeps working unchanged.
//
// Kept at the global namespace to match GPUConfig / GPUContext
// (defined globally in ptxsim/gpu_context.h).
struct InstructionLatencyConfig {
    int ld_global_cycles = -1;
    bool ld_global_long_delay = true;

    int st_global_cycles = -1;
    bool st_global_long_delay = false;

    int mul_cycles = -1;
    bool mul_long_delay = false;

    int div_cycles = -1;
    bool div_long_delay = true;

    int bar_sync_cycles = -1;
    bool bar_sync_long_delay = false;

    int default_cycles = -1;
    bool default_long_delay = false;
};

#endif