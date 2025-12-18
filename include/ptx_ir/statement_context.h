#ifndef STATEMENT_CONTEXT_H
#define STATEMENT_CONTEXT_H

#include "operand_context.h"
#include <vector>

class StatementContext {
public:
    StatementType statementType;
    void *statement;

    struct REG {
        int regNum = 1;
        std::vector<Qualifier> regDataType;
        std::string regName;
    };

    struct CONST {
        int constAlign = 0;
        int constSize = 1;
        std::vector<Qualifier> constDataType;
        std::string constName;
    };

    struct SHARED {
        int sharedAlign = 0;
        int sharedSize = 1;
        std::vector<Qualifier> sharedDataType;
        std::string sharedName;
    };

    struct LOCAL {
        int localAlign = 0;
        int localSize = 1;
        std::vector<Qualifier> localDataType;
        std::string localName;
    };

    struct DOLLOR {
        std::string dollorName;
    };

    struct AT {
        OperandContext atPred;
        std::string atLabelName;
    };

    struct PRAGMA {
        std::string pragmaString;
    };

    struct RET {};

    struct BAR {
        std::vector<Qualifier> braQualifier;
        std::string barType;
        int barId;
    };

    struct BRA {
        std::vector<Qualifier> braQualifier;
        std::string braTarget;
    };

    struct RCP {
        std::vector<Qualifier> rcpQualifier;
        OperandContext rcpOp[2];
    };

    struct LD {
        std::vector<Qualifier> ldQualifier;
        OperandContext ldOp[2];
    };

    struct MOV {
        std::vector<Qualifier> movQualifier;
        OperandContext movOp[2];
    };

    struct SETP {
        std::vector<Qualifier> setpQualifier;
        OperandContext setpOp[3];
    };

    struct CVTA {
        std::vector<Qualifier> cvtaQualifier;
        OperandContext cvtaOp[2];
    };

    struct CVT {
        std::vector<Qualifier> cvtQualifier;
        OperandContext cvtOp[2];
    };

    struct MUL {
        std::vector<Qualifier> mulQualifier;
        OperandContext mulOp[3];
    };

    struct MUL24 {
        std::vector<Qualifier> mul24Qualifier;
        OperandContext mul24Op[3];
    };

    struct DIV {
        std::vector<Qualifier> divQualifier;
        OperandContext divOp[3];
    };

    struct SUB {
        std::vector<Qualifier> subQualifier;
        OperandContext subOp[3];
    };

    struct ADD {
        std::vector<Qualifier> addQualifier;
        OperandContext addOp[3];
    };

    struct SHL {
        std::vector<Qualifier> shlQualifier;
        OperandContext shlOp[3];
    };

    struct SHR {
        std::vector<Qualifier> shrQualifier;
        OperandContext shrOp[3];
    };

    struct MAX {
        std::vector<Qualifier> maxQualifier;
        OperandContext maxOp[3];
    };

    struct MIN {
        std::vector<Qualifier> minQualifier;
        OperandContext minOp[3];
    };

    struct AND {
        std::vector<Qualifier> andQualifier;
        OperandContext andOp[3];
    };

    struct OR {
        std::vector<Qualifier> orQualifier;
        OperandContext orOp[3];
    };

    struct ST {
        std::vector<Qualifier> stQualifier;
        OperandContext stOp[2];
    };

    struct SELP {
        std::vector<Qualifier> selpQualifier;
        OperandContext selpOp[4];
    };

    struct MAD {
        std::vector<Qualifier> madQualifier;
        OperandContext madOp[4];
    };

    struct MAD24 {
        std::vector<Qualifier> mad24Qualifier;
        OperandContext mad24Op[4];
    };

    struct FMA {
        std::vector<Qualifier> fmaQualifier;
        OperandContext fmaOp[4];
    };

    struct WMMA {
        WmmaType wmmaType;
        std::vector<Qualifier> wmmaQualifier;
        OperandContext wmmaOp[4];
    };

    struct NEG {
        std::vector<Qualifier> negQualifier;
        OperandContext negOp[2];
    };

    struct NOT {
        std::vector<Qualifier> notQualifier;
        OperandContext notOp[2];
    };

    struct SQRT {
        std::vector<Qualifier> sqrtQualifier;
        OperandContext sqrtOp[2];
    };

    struct COS {
        std::vector<Qualifier> cosQualifier;
        OperandContext cosOp[2];
    };

    struct LG2 {
        std::vector<Qualifier> lg2Qualifier;
        OperandContext lg2Op[2];
    };

    struct EX2 {
        std::vector<Qualifier> ex2Qualifier;
        OperandContext ex2Op[2];
    };

    struct ATOM {
        std::vector<Qualifier> atomQualifier;
        OperandContext atomOp[4];
        int operandNum = 0;
    };

    struct XOR {
        std::vector<Qualifier> xorQualifier;
        OperandContext xorOp[3];
    };

    struct ABS {
        std::vector<Qualifier> absQualifier;
        OperandContext absOp[2];
    };

    struct SIN {
        std::vector<Qualifier> sinQualifier;
        OperandContext sinOp[2];
    };

    struct RSQRT {
        std::vector<Qualifier> rsqrtQualifier;
        OperandContext rsqrtOp[2];
    };

#define DEFINE_STATEMENT_STRUCT(Name, OpCount)                                 \
    struct Name {                                                              \
        static constexpr int op_count = OpCount;                               \
        std::vector<Qualifier> qualifier;                                      \
        OperandContext op[OpCount];                                            \
    }

    DEFINE_STATEMENT_STRUCT(POPC, 2);
    DEFINE_STATEMENT_STRUCT(CLZ, 2);

    struct REM {
        std::vector<Qualifier> remQualifier;
        OperandContext remOp[3];
    };

    StatementContext() : statementType(S_REG), statement(nullptr) {}
    ~StatementContext();

    // 深拷贝方法
    StatementContext(const StatementContext &other);
};

#endif // STATEMENT_CONTEXT_H