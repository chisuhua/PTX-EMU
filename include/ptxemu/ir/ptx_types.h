#ifndef PTXEMU_IR_PTX_TYPES_H
#define PTXEMU_IR_PTX_TYPES_H

#include <cassert>
#include <string>

// Phase 1 (HSK-8 ack 738b412c): namespace wrapping for public IR types.
// ptx_types.h content moved verbatim from include/ptx_ir/ptx_types.h,
// wrapped in ptxemu::ir namespace. ptx_ir::Qualifier etc. aliased via
// forwarding header include/ptx_ir/ptx_types.h (one release cycle).

namespace ptxemu {
namespace ir {

void extractREG(std::string s, int &idx, std::string &name);

enum class Qualifier {
#define X(enum_val, ...) enum_val,
#include "ptx_qualifier.def"
#undef X
    Q_UNKNOWN
};

std::string Q2s(Qualifier q);
int Q2bytes(Qualifier q);

enum StatementType {
#define X(enum_val, struct_name, str, opcount, _, instr_kind) enum_val,
#include "ptx_op.def"
#undef X
    S_UNKNOWN
};

std::string S2s(StatementType s);

enum OperandType { O_REG, O_VAR, O_IMM, O_VEC, O_FA, O_PRED };

enum class OperandKind { REG, VAR, IMM, VEC, ADDR, PRED };

}  // namespace ir
}  // namespace ptxemu

#endif  // PTXEMU_IR_PTX_TYPES_H