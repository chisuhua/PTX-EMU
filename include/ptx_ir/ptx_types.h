#ifndef PTX_TYPES_H
#define PTX_TYPES_H

#include <ptxemu/ir/ptx_types.h>

namespace ptx_ir = ::ptxemu::ir;

using ::ptxemu::ir::Qualifier;
using ::ptxemu::ir::StatementType;
using ::ptxemu::ir::OperandType;
using ::ptxemu::ir::OperandKind;

// Temporary enum: bridge (1.5c+d only — removed in 1.5k after caller
// sweeps complete). C++20 'using enum' injects unscoped enumerators
// into the global namespace so the ~87 caller files that still
// reference S_*/O_* directly keep compiling through 1.5e-1.5i3.
using enum ::ptxemu::ir::StatementType;
using enum ::ptxemu::ir::OperandType;

#endif  // PTX_TYPES_H
