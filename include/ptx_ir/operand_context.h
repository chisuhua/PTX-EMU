#ifndef OPERAND_CONTEXT_H
#define OPERAND_CONTEXT_H

#include "ptx_types.h"
#include <vector>

class OperandContext {
public:
    OperandType operandType;
    void *operand = nullptr;

    struct REG {
        std::string regName;
        int regIdx;
    };

    struct VAR {
        std::string varName;
    };

    struct IMM {
        std::string immVal;
    };

    struct VEC {
        std::vector<OperandContext> vec;
    };

    struct FA { // fetch address
        enum BaseType { CONSTANT, SHARED };
        enum OffsetType { IMMEDIATE, REGISTER };

        BaseType baseType;
        std::string baseName;
        OffsetType offsetType;
        std::string offsetVal;
        std::string ID;
        OperandContext *reg;
    };

    struct PRED {
        bool isNot{false};
        OperandContext *pred;
    };

    OperandContext() : operandType(O_REG), operand(nullptr) {}
    ~OperandContext();

    // 深拷贝方法
    OperandContext(const OperandContext &other);
    OperandContext &operator=(const OperandContext &other);
};

#endif // OPERAND_CONTEXT_H