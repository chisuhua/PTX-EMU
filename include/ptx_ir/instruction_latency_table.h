#ifndef INSTRUCTION_LATENCY_TABLE_H
#define INSTRUCTION_LATENCY_TABLE_H

#include "ptx_ir/instruction_latency.h"
#include "ptx_ir/instruction_latency_config.h"
#include "ptx_ir/ptx_types.h"   // for StatementType
#include <map>
#include <cstdint>

namespace ptxsim {

// Per-instruction-type latency lookup.
//
// Implementation note: a separate side-table was chosen over extending
// the ptx_op.def X-Macro with a latency field, because the X-Macro is
// included in 5 different translation units (ptx_types.h, statement_context.cpp,
// ptx_parser.h, ptx_visiter.h, instruction_handlers.h, ptx_parser.cpp),
// each of which expects exactly 5 arguments in a fixed order. Adding a 6th
// field would require updating every expansion site and is high risk.
//
// Latencies are sourced from the active GPU JSON config (see
// gpu_context.h::GPUConfig::InstructionLatencyConfig) and override the
// constexpr defaults in instruction_latency.h. Values are loaded at
// GPUContext construction so getLatency() is a single map lookup.
//
// To extend the table with a new instruction type, add a case in
// resolve_default() AND a matching entry in InstructionLatencyConfig.

class InstructionLatencyTable {
public:
    static InstructionLatencyTable& instance();

    // Replace values from `cfg`. Entries with cycles <= 0 are left at
    // their current value (default unless previously overridden).
    void load(const InstructionLatencyConfig& cfg);

    // Reset all entries to constexpr defaults from instruction_latency.h.
    void reset_to_defaults();

    InstructionLatency get(ptxemu::ir::StatementType type) const;

private:
    InstructionLatencyTable();
    InstructionLatency resolve_default(ptxemu::ir::StatementType type) const;

    std::map<ptxemu::ir::StatementType, InstructionLatency> table_;
};

// Backward-compatible free function. Equivalent to
// InstructionLatencyTable::instance().get(type).
InstructionLatency getLatency(ptxemu::ir::StatementType type);

} // namespace ptxsim

#endif
