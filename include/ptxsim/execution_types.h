#ifndef EXECUTION_TYPES_H
#define EXECUTION_TYPES_H

#include <cstdint>
#include <sstream>
#include <string>

enum EXE_STATE { RUN, EXIT, BAR };

struct Dim3 {
    uint32_t x, y, z;
    Dim3(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) : x(x), y(y), z(z) {}

    std::string to_string() const {
        std::ostringstream oss;
        oss << "[" << x << "," << y << "," << z << "]";
        return oss.str();
    }
};

/*
enum StatementType {
    S_REG,
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
    S_WMMA,
    S_CONST
};
enum class Qualifier {
    Q_S64,
    Q_U64,
    Q_B64,
    Q_S32,
    Q_U32,
    Q_B32,
    Q_S16,
    Q_U16,
    Q_B16,
    Q_S8,
    Q_U8,
    Q_B8,
    Q_F64,
    Q_F32,
    Q_F16,
    Q_F8,
    Q_PRED,
    Q_WIDE,
    Q_HI,
    Q_LO,
    Q_SAT,
    Q_SHARED,
    Q_GLOBAL,
    Q_LOCAL,
    Q_CONST,
    Q_V2,
    Q_V4,
    Q_ROW,
    Q_COL,
    Q_EQ,
    Q_NE,
    Q_LT,
    Q_LE,
    Q_GT,
    Q_GE,
    Q_LTU,
    Q_LEU,
    Q_GEU,
    Q_NEU,
    Q_GTU,
    Q_DOTADD
};
*/

#endif // EXECUTION_TYPES_H