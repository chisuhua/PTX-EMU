#ifndef OPERAND_CONTEXT_H
#define OPERAND_CONTEXT_H

#include "ptx_types.h"
#include <string>
#include <vector>

class OperandContext {
public:
    OperandType operandType;
    void *operand = nullptr;
    void *operand_phy_addr = nullptr; // use exeute state to fetech real operand

    struct REG {
        std::string regName;
        int regIdx;

        // 添加获取完整寄存器名称的方法
        std::string getFullName() const {
            return regName + std::to_string(regIdx);
        }
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

    explicit OperandContext(OperandType operand_type, void *operand)
        : operandType(operand_type), operand(operand) {}
    OperandContext()
        : operandType(O_REG), operand(nullptr), operand_phy_addr(nullptr) {}
    ~OperandContext();

    // 深拷贝方法
    OperandContext(const OperandContext &other);
    OperandContext &operator=(const OperandContext &other);

    // 添加toString方法用于获取操作数的字符串表示
    std::string toString() const;
};

#endif // OPERAND_CONTEXT_H