#include "ptx_ir/operand_context.h"
#include <cstdint>
#include <sstream>

std::string OperandContext::toString(int bytes) const {
    std::ostringstream oss;

    switch (operandType) {
    case O_REG: {
        auto *reg = static_cast<REG *>(operand);
        if (reg) {
            oss << "%" << reg->regName;
            if (reg->regIdx >= 0) {
                oss << reg->regIdx;
            }
        }
        break;
    }
    case O_VAR: {
        auto *var = static_cast<VAR *>(operand);
        if (var) {
            oss << var->varName;
        }
        break;
    }
    case O_IMM: {
        auto *imm = static_cast<IMM *>(operand);
        if (imm) {
            oss << imm->immVal;
        }
        break;
    }
    case O_VEC: {
        auto *vec = static_cast<VEC *>(operand);
        if (vec) {
            oss << "{";
            for (size_t i = 0; i < vec->vec.size(); ++i) {
                if (i > 0)
                    oss << ", ";
                oss << vec->vec[i].toString();
            }
            oss << "}";
        }
        break;
    }
    case O_FA: {
        auto *fa = static_cast<FA *>(operand);
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
        auto *pred = static_cast<PRED *>(operand);
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
    oss << " phy_addr:" << std::hex << operand_phy_addr;
    if (operand_phy_addr != nullptr && bytes == 1) {
        oss << " value:0x" << std::hex << (int)*(uint8_t *)operand_phy_addr << std::dec;
    } else if (operand_phy_addr != nullptr && bytes == 2) {
        oss << " value:0x" << std::hex << *(uint16_t *)operand_phy_addr << std::dec;
    } else if (operand_phy_addr != nullptr && bytes == 4) {
        oss << " value:0x" << std::hex << *(uint32_t *)operand_phy_addr << std::dec;
    } else if (operand_phy_addr != nullptr && bytes == 8) {
        oss << " value:0x" << std::hex << *(uint64_t *)operand_phy_addr << std::dec;
    }

    return oss.str();
}

// Adding OperandContext destructor implementation
OperandContext::~OperandContext() {
    // Free memory for the operand based on operandType
    switch (operandType) {
    case O_REG:
        if (operand) {
            delete static_cast<OperandContext::REG *>(operand);
        }
        break;
    case O_VEC:
        if (operand) {
            delete static_cast<OperandContext::VEC *>(operand);
        }
        break;
    case O_FA:
        if (operand) {
            auto fa = static_cast<OperandContext::FA *>(operand);
            // Note: Need to free the object pointed to by reg first
            if (fa->reg) {
                delete fa->reg;
                fa->reg = nullptr; // Prevent dangling pointer
            }
            delete fa;
        }
        break;
    case O_PRED:
        if (operand) {
            auto pred = static_cast<OperandContext::PRED *>(operand);
            // Note: Need to free the object pointed to by pred first
            if (pred->pred) {
                delete pred->pred;
                pred->pred = nullptr; // Prevent dangling pointer
            }
            delete pred;
        }
        break;
    case O_IMM:
        if (operand) {
            delete static_cast<OperandContext::IMM *>(operand);
        }
        break;
    case O_VAR:
        if (operand) {
            delete static_cast<OperandContext::VAR *>(operand);
        }
        break;
    }
    operand = nullptr;
}

// Adding OperandContext copy constructor implementation
OperandContext::OperandContext(const OperandContext &other)
    : operandType(other.operandType), operand(nullptr),
      operand_phy_addr(nullptr) {
    switch (operandType) {
    case O_REG:
        if (other.operand) {
            auto reg = new REG();
            auto other_reg = static_cast<const REG *>(other.operand);
            reg->regName = other_reg->regName;
            reg->regIdx = other_reg->regIdx;
            operand = reg;
        }
        break;
    case O_VEC:
        if (other.operand) {
            auto vec = new VEC();
            auto other_vec = static_cast<const VEC *>(other.operand);
            vec->vec = other_vec->vec;
            operand = vec;
        }
        break;
    case O_FA:
        if (other.operand) {
            auto fa = new FA();
            auto other_fa = static_cast<const FA *>(other.operand);
            fa->baseType = other_fa->baseType;
            fa->baseName = other_fa->baseName;
            fa->offsetType = other_fa->offsetType;
            fa->offsetVal = other_fa->offsetVal;
            fa->ID = other_fa->ID;

            // Deep copy reg pointer
            if (other_fa->reg) {
                fa->reg = new OperandContext(*(other_fa->reg));
            } else {
                fa->reg = nullptr;
            }

            operand = fa;
        }
        break;
    case O_PRED:
        if (other.operand) {
            auto pred = new PRED();
            auto other_pred = static_cast<const PRED *>(other.operand);
            pred->isNot = other_pred->isNot;

            // Deep copy pred pointer
            if (other_pred->pred) {
                pred->pred = new OperandContext(*(other_pred->pred));
            } else {
                pred->pred = nullptr;
            }

            operand = pred;
        }
        break;
    case O_IMM:
        if (other.operand) {
            auto imm = new IMM();
            auto other_imm = static_cast<const IMM *>(other.operand);
            imm->immVal = other_imm->immVal;
            operand = imm;
        }
        break;
    case O_VAR:
        if (other.operand) {
            auto var = new VAR();
            auto other_var = static_cast<const VAR *>(other.operand);
            var->varName = other_var->varName;
            operand = var;
        }
        break;
    }
}

// Adding OperandContext assignment operator implementation
OperandContext &OperandContext::operator=(const OperandContext &other) {
    // Self-assignment check
    if (this == &other) {
        return *this;
    }

    // Free current resources first
    switch (operandType) {
    case O_REG:
        if (operand) {
            delete static_cast<REG *>(operand);
        }
        break;
    case O_VEC:
        if (operand) {
            delete static_cast<VEC *>(operand);
        }
        break;
    case O_FA:
        if (operand) {
            auto fa = static_cast<FA *>(operand);
            if (fa->reg) {
                delete fa->reg;
            }
            delete fa;
        }
        break;
    case O_PRED:
        if (operand) {
            auto pred = static_cast<PRED *>(operand);
            if (pred->pred) {
                delete pred->pred;
            }
            delete pred;
        }
        break;
    case O_IMM:
        if (operand) {
            delete static_cast<IMM *>(operand);
        }
        break;
    case O_VAR:
        if (operand) {
            delete static_cast<VAR *>(operand);
        }
        break;
    }

    // Update type
    operandType = other.operandType;

    // Deep copy new content
    switch (operandType) {
    case O_REG:
        if (other.operand) {
            auto reg = new REG();
            auto other_reg = static_cast<const REG *>(other.operand);
            reg->regName = other_reg->regName;
            reg->regIdx = other_reg->regIdx;
            operand = reg;
        } else {
            operand = nullptr;
        }
        break;
    case O_VEC:
        if (other.operand) {
            auto vec = new VEC();
            auto other_vec = static_cast<const VEC *>(other.operand);
            vec->vec = other_vec->vec;
            operand = vec;
        } else {
            operand = nullptr;
        }
        break;
    case O_FA:
        if (other.operand) {
            auto fa = new FA();
            auto other_fa = static_cast<const FA *>(other.operand);
            fa->baseType = other_fa->baseType;
            fa->baseName = other_fa->baseName;
            fa->offsetType = other_fa->offsetType;
            fa->offsetVal = other_fa->offsetVal;
            fa->ID = other_fa->ID;

            // Deep copy reg pointer
            if (other_fa->reg) {
                fa->reg = new OperandContext(*(other_fa->reg));
            } else {
                fa->reg = nullptr;
            }

            operand = fa;
        } else {
            operand = nullptr;
        }
        break;
    case O_PRED:
        if (other.operand) {
            auto pred = new PRED();
            auto other_pred = static_cast<const PRED *>(other.operand);
            pred->isNot = other_pred->isNot;

            // Deep copy pred pointer
            if (other_pred->pred) {
                pred->pred = new OperandContext(*(other_pred->pred));
            } else {
                pred->pred = nullptr;
            }

            operand = pred;
        } else {
            operand = nullptr;
        }
        break;
    case O_IMM:
        if (other.operand) {
            auto imm = new IMM();
            auto other_imm = static_cast<const IMM *>(other.operand);
            imm->immVal = other_imm->immVal;
            operand = imm;
        } else {
            operand = nullptr;
        }
        break;
    case O_VAR:
        if (other.operand) {
            auto var = new VAR();
            auto other_var = static_cast<const VAR *>(other.operand);
            var->varName = other_var->varName;
            operand = var;
        } else {
            operand = nullptr;
        }
        break;
    }

    return *this;
}
