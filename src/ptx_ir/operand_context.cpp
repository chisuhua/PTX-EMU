#include "ptx_ir/operand_context.h"
#include <sstream>

std::string OperandContext::toString() const {
    std::ostringstream oss;
    
    switch (operandType) {
        case O_REG: {
            auto* reg = static_cast<REG*>(operand);
            if (reg) {
                oss << "%" << reg->regName;
                if (reg->regIdx >= 0) {
                    oss << reg->regIdx;
                }
            }
            break;
        }
        case O_VAR: {
            auto* var = static_cast<VAR*>(operand);
            if (var) {
                oss << var->varName;
            }
            break;
        }
        case O_IMM: {
            auto* imm = static_cast<IMM*>(operand);
            if (imm) {
                oss << imm->immVal;
            }
            break;
        }
        case O_VEC: {
            auto* vec = static_cast<VEC*>(operand);
            if (vec) {
                oss << "{";
                for (size_t i = 0; i < vec->vec.size(); ++i) {
                    if (i > 0) oss << ", ";
                    oss << vec->vec[i].toString();
                }
                oss << "}";
            }
            break;
        }
        case O_FA: {
            auto* fa = static_cast<FA*>(operand);
            if (fa) {
                oss << "[" << fa->baseName;
                if (fa->offsetType == FA::IMMEDIATE) {
                    oss << " + " << fa->offsetVal;
                } else if (fa->offsetType == FA::REGISTER && fa->reg) {
                    oss << " + " << fa->reg->toString();
                }
                oss << "]";
            }
            break;
        }
        case O_PRED: {
            auto* pred = static_cast<PRED*>(operand);
            if (pred) {
                if (pred->isNot) {
                    oss << "!";
                }
                if (pred->pred) {
                    oss << pred->pred->toString();
                }
            }
            break;
        }
        default:
            oss << "<unknown>";
            break;
    }
    
    return oss.str();
}