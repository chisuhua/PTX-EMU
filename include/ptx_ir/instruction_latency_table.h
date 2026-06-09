#ifndef INSTRUCTION_LATENCY_TABLE_H
#define INSTRUCTION_LATENCY_TABLE_H

#include "instruction_latency.h"
#include "ptx_types.h"   // for StatementType
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
// Instead, this table maps StatementType -> InstructionLatency and is
// queried at runtime. To extend, add a case in getLatency() below.

InstructionLatency getLatency(StatementType type);

} // namespace ptxsim

#endif
