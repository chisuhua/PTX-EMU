#ifndef PTX_TYPES_H
#define PTX_TYPES_H

#include <cassert>
#include <string>

void extractREG(std::string s, int &idx, std::string &name);

enum class Qualifier {
    Q_U64,
    Q_U32,
    Q_U16,
    Q_U8,
    Q_PRED,
    Q_B8,
    Q_B16,
    Q_B32,
    Q_B64,
    Q_F8,
    Q_F16,
    Q_F32,
    Q_F64,
    Q_S8,
    Q_S16,
    Q_S32,
    Q_S64,
    Q_V2,
    Q_V4,
    Q_CONST,
    Q_PARAM,
    Q_GLOBAL,
    Q_LOCAL,
    Q_SHARED,
    Q_GT,
    Q_GE,
    Q_EQ,
    Q_NE,
    Q_LT,
    Q_TO,
    Q_WIDE,
    Q_SYNC,
    Q_LO,
    Q_HI,
    Q_UNI,
    Q_RN,
    Q_A,
    Q_B,
    Q_D,
    Q_ROW,
    Q_ALIGNED,
    Q_M8N8K4,
    Q_M16N16K16,
    Q_NEU,
    Q_NC,
    Q_FTZ,
    Q_APPROX,
    Q_LTU,
    Q_LE,
    Q_GTU,
    Q_LEU,
    Q_DOTADD,
    Q_GEU,
    Q_RZI,
    Q_DOTOR,
    Q_SAT,
    S_UNKNOWN
};

std::string Q2s(Qualifier q);
int Q2bytes(Qualifier q);

enum StatementType {
    S_REG,
    S_CONST,
    S_SHARED,
    S_LOCAL,
    S_DOLLOR,
    S_AT,
    S_PRAGMA,
    S_RET,
    S_BAR,
    S_BRA,
    S_RCP,
    S_LD,
    S_MOV,
    S_SETP,
    S_CVTA,
    S_CVT,
    S_MUL,
    S_DIV,
    S_SUB,
    S_ADD,
    S_SHL,
    S_SHR,
    S_MAX,
    S_MIN,
    S_AND,
    S_OR,
    S_ST,
    S_SELP,
    S_MAD,
    S_FMA,
    S_WMMA,
    S_NEG,
    S_NOT,
    S_SQRT,
    S_COS,
    S_LG2,
    S_EX2,
    S_ATOM,
    S_XOR,
    S_ABS,
    S_SIN,
    S_REM,
    S_RSQRT,
    S_UNKNOWN
};

std::string S2s(StatementType s);

enum OperandType { O_REG, O_VAR, O_IMM, O_VEC, O_FA, O_PRED };

enum WmmaType { WMMA_LOAD, WMMA_STORE, WMMA_MMA };

// enum EXE_STATE { RUN, BAR, EXIT };

#endif // PTX_TYPES_H