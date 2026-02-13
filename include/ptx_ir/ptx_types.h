#ifndef PTX_TYPES_H
#define PTX_TYPES_H

#include <cassert>
#include <string>

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

enum WmmaType { WMMA_LOAD, WMMA_STORE, WMMA_MMA };

enum class OperandKind { REG, VAR, IMM, VEC, ADDR, PRED };

#endif // PTX_TYPES_H
