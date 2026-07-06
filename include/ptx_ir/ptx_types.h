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
    // Blackwell Tensor Core Generator (PTX ISA §9.7.16, sm_100+) — ADR-0016.
    // Added outside the X-Macro loop because the tcgen05.* grammar has a
    // single tcgen05Inst rule, not 11 per-sub-op rules; the per-instruction
    // X-Macro expansion in ptx_visitor.cpp / instruction_handlers.cpp would
    // fail to find matching ANTLR Context classes. See design.md Decision 1–5.
    S_TCGEN05_ALLOC,
    S_TCGEN05_DEALLOC,
    S_TCGEN05_RELINQUISH,
    S_TCGEN05_LD,
    S_TCGEN05_ST,
    S_TCGEN05_CP,
    S_TCGEN05_MMA,
    S_TCGEN05_MMA_WS,
    S_TCGEN05_COMMIT,
    S_TCGEN05_WAIT,
    S_TCGEN05_FENCE,
    S_UNKNOWN
};

std::string S2s(StatementType s);

enum OperandType { O_REG, O_VAR, O_IMM, O_VEC, O_FA, O_PRED };

enum WmmaType { WMMA_LOAD, WMMA_STORE, WMMA_MMA, WMMA_COMMIT, WMMA_WAIT };

enum class OperandKind { REG, VAR, IMM, VEC, ADDR, PRED };

#endif // PTX_TYPES_H
