#include "ptx_ir/ptx_types.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"
#include <cmath>
#include <cstdlib>
#include <string>

void extractREG(std::string s, int &idx, std::string &name) {
    // Handle special registers like %tid.x, %ctaid.x, etc.
    size_t dotPos = s.find('.');
    if (dotPos != std::string::npos) {
        name = s;  // Keep the full name including the dot part
        idx = 0;
        return;
    }
    
    // Handle regular registers like %r1, %rd2, etc.
    int ret = 0;
    for (char c : s) {
        if (c >= '0' && c <= '9') {
            ret = ret * 10 + c - '0';
        }
    }
    idx = ret;
    for (int i = 0; i < s.size(); i++) {
        if ((s[i] >= '0' && s[i] <= '9')) {
            name = s.substr(0, i);
            return;
        }
    }
    name = s;
}

std::string Q2s(Qualifier q) {
    switch (q) {
    case Qualifier::Q_U64:
        return ".u64";
    case Qualifier::Q_U32:
        return ".u32";
    case Qualifier::Q_U16:
        return ".u16";
    case Qualifier::Q_U8:
        return ".u8";
    case Qualifier::Q_PRED:
        return ".pred";
    case Qualifier::Q_B8:
        return ".b8";
    case Qualifier::Q_B16:
        return ".b16";
    case Qualifier::Q_B32:
        return ".b32";
    case Qualifier::Q_B64:
        return ".b64";
    case Qualifier::Q_F8:
        return ".f8";
    case Qualifier::Q_F16:
        return ".f16";
    case Qualifier::Q_F32:
        return ".f32";
    case Qualifier::Q_F64:
        return ".f64";
    case Qualifier::Q_S8:
        return ".s8";
    case Qualifier::Q_S16:
        return ".s16";
    case Qualifier::Q_S32:
        return ".s32";
    case Qualifier::Q_S64:
        return ".s64";
    case Qualifier::Q_V2:
        return ".v2";
    case Qualifier::Q_V4:
        return ".v4";
    case Qualifier::Q_CONST:
        return ".const";
    case Qualifier::Q_PARAM:
        return ".param";
    case Qualifier::Q_GLOBAL:
        return ".global";
    case Qualifier::Q_LOCAL:
        return ".local";
    case Qualifier::Q_SHARED:
        return ".shared";
    case Qualifier::Q_GT:
        return ".gt";
    case Qualifier::Q_GE:
        return ".ge";
    case Qualifier::Q_EQ:
        return ".eq";
    case Qualifier::Q_NE:
        return ".ne";
    case Qualifier::Q_LT:
        return ".lt";
    case Qualifier::Q_TO:
        return ".to";
    case Qualifier::Q_WIDE:
        return ".wide";
    case Qualifier::Q_SYNC:
        return ".sync";
    case Qualifier::Q_LO:
        return ".lo";
    case Qualifier::Q_HI:
        return ".hi";
    case Qualifier::Q_UNI:
        return ".uni";
    case Qualifier::Q_RN:
        return ".rn";
    case Qualifier::Q_A:
        return ".a";
    case Qualifier::Q_B:
        return ".b";
    case Qualifier::Q_D:
        return ".d";
    case Qualifier::Q_ROW:
        return ".row";
    case Qualifier::Q_ALIGNED:
        return ".aligned";
    case Qualifier::Q_M8N8K4:
        return ".m8n8k4";
    case Qualifier::Q_M16N16K16:
        return ".m16n16k16";
    case Qualifier::Q_NEU:
        return ".neu";
    case Qualifier::Q_NC:
        return ".nc";
    case Qualifier::Q_FTZ:
        return ".ftz";
    case Qualifier::Q_APPROX:
        return ".approx";
    case Qualifier::Q_LTU:
        return ".ltu";
    case Qualifier::Q_LE:
        return ".le";
    case Qualifier::Q_GTU:
        return ".gtu";
    case Qualifier::Q_LEU:
        return ".leu";
    case Qualifier::Q_DOTADD:
        return ".add";
    case Qualifier::Q_GEU:
        return ".geu";
    case Qualifier::Q_RZI:
        return ".rzi";
    case Qualifier::Q_DOTOR:
        return ".or";
    case Qualifier::Q_SAT:
        return ".sat";
    default:
        assert(0 && "Unsupported qualifier");
        return "";
    }
}

int Q2bytes(Qualifier q) {
    switch (q) {
    case Qualifier::Q_U8:
    case Qualifier::Q_B8:
    case Qualifier::Q_S8:
    case Qualifier::Q_F8:
        return 1;
    case Qualifier::Q_U16:
    case Qualifier::Q_B16:
    case Qualifier::Q_S16:
    case Qualifier::Q_F16:
        return 2;
    case Qualifier::Q_U32:
    case Qualifier::Q_B32:
    case Qualifier::Q_S32:
    case Qualifier::Q_F32:
        return 4;
    case Qualifier::Q_U64:
    case Qualifier::Q_B64:
    case Qualifier::Q_S64:
    case Qualifier::Q_F64:
        return 8;
    case Qualifier::Q_PRED:
        return 1; // Assume predicate is 1 byte
    default:
        assert(0 && "Unsupported qualifier for byte calculation");
        return 4; // Default return 4 bytes
    }
}

std::string S2s(StatementType s) {
    switch (s) {
    case S_REG:
        return "reg";
    case S_CONST:
        return "const";
    case S_SHARED:
        return "shared";
    case S_LOCAL:
        return "local";
    case S_DOLLOR:
        return "$";
    case S_AT:
        return "@";
    case S_PRAGMA:
        return "pragma";
    case S_RET:
        return "ret";
    case S_BAR:
        return "bar";
    case S_BRA:
        return "bra";
    case S_RCP:
        return "rcp";
    case S_LD:
        return "ld";
    case S_MOV:
        return "mov";
    case S_SETP:
        return "setp";
    case S_CVTA:
        return "cvta";
    case S_CVT:
        return "cvt";
    case S_MUL:
        return "mul";
    case S_DIV:
        return "div";
    case S_SUB:
        return "sub";
    case S_ADD:
        return "add";
    case S_SHL:
        return "shl";
    case S_SHR:
        return "shr";
    case S_MAX:
        return "max";
    case S_MIN:
        return "min";
    case S_AND:
        return "and";
    case S_OR:
        return "or";
    case S_ST:
        return "st";
    case S_SELP:
        return "selp";
    case S_MAD:
        return "mad";
    case S_FMA:
        return "fma";
    case S_WMMA:
        return "wmma";
    case S_NEG:
        return "neg";
    case S_NOT:
        return "not";
    case S_SQRT:
        return "sqrt";
    case S_COS:
        return "cos";
    case S_LG2:
        return "lg2";
    case S_EX2:
        return "ex2";
    case S_ATOM:
        return "atom";
    case S_XOR:
        return "xor";
    case S_ABS:
        return "abs";
    case S_SIN:
        return "sin";
    case S_REM:
        return "rem";
    case S_RSQRT:
        return "rsqrt";
    case S_UNKNOWN:
        return "unknown";
    default:
        assert(0 && "Unsupported statement type");
        return "";
    }
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
                fa->reg = nullptr;  // Prevent dangling pointer
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
                pred->pred = nullptr;  // Prevent dangling pointer
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
    : operandType(other.operandType), operand(nullptr) {
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

// Adding StatementContext destructor implementation
StatementContext::~StatementContext() {
    // Free memory for the statement based on statementType
    switch (statementType) {
    case S_REG:
        if (statement) {
            delete static_cast<StatementContext::REG*>(statement);
        }
        break;
    case S_CONST:
        if (statement) {
            delete static_cast<StatementContext::CONST*>(statement);
        }
        break;
    case S_SHARED:
        if (statement) {
            delete static_cast<StatementContext::SHARED*>(statement);
        }
        break;
    case S_LOCAL:
        if (statement) {
            delete static_cast<StatementContext::LOCAL*>(statement);
        }
        break;
    case S_DOLLOR:
        if (statement) {
            delete static_cast<StatementContext::DOLLOR*>(statement);
        }
        break;
    case S_AT:
        if (statement) {
            delete static_cast<StatementContext::AT*>(statement);
        }
        break;
    case S_PRAGMA:
        if (statement) {
            delete static_cast<StatementContext::PRAGMA*>(statement);
        }
        break;
    case S_RET:
        if (statement) {
            delete static_cast<StatementContext::RET*>(statement);
        }
        break;
    case S_BAR:
        if (statement) {
            delete static_cast<StatementContext::BAR*>(statement);
        }
        break;
    case S_BRA:
        if (statement) {
            delete static_cast<StatementContext::BRA*>(statement);
        }
        break;
    case S_RCP:
        if (statement) {
            delete static_cast<StatementContext::RCP*>(statement);
        }
        break;
    case S_LD:
        if (statement) {
            delete static_cast<StatementContext::LD*>(statement);
        }
        break;
    case S_MOV:
        if (statement) {
            delete static_cast<StatementContext::MOV*>(statement);
        }
        break;
    case S_SETP:
        if (statement) {
            delete static_cast<StatementContext::SETP*>(statement);
        }
        break;
    case S_CVTA:
        if (statement) {
            delete static_cast<StatementContext::CVTA*>(statement);
        }
        break;
    case S_CVT:
        if (statement) {
            delete static_cast<StatementContext::CVT*>(statement);
        }
        break;
    case S_MUL:
        if (statement) {
            delete static_cast<StatementContext::MUL*>(statement);
        }
        break;
    case S_DIV:
        if (statement) {
            delete static_cast<StatementContext::DIV*>(statement);
        }
        break;
    case S_SUB:
        if (statement) {
            delete static_cast<StatementContext::SUB*>(statement);
        }
        break;
    case S_ADD:
        if (statement) {
            delete static_cast<StatementContext::ADD*>(statement);
        }
        break;
    case S_SHL:
        if (statement) {
            delete static_cast<StatementContext::SHL*>(statement);
        }
        break;
    case S_SHR:
        if (statement) {
            delete static_cast<StatementContext::SHR*>(statement);
        }
        break;
    case S_MAX:
        if (statement) {
            delete static_cast<StatementContext::MAX*>(statement);
        }
        break;
    case S_MIN:
        if (statement) {
            delete static_cast<StatementContext::MIN*>(statement);
        }
        break;
    case S_AND:
        if (statement) {
            delete static_cast<StatementContext::AND*>(statement);
        }
        break;
    case S_OR:
        if (statement) {
            delete static_cast<StatementContext::OR*>(statement);
        }
        break;
    case S_ST:
        if (statement) {
            delete static_cast<StatementContext::ST*>(statement);
        }
        break;
    case S_SELP:
        if (statement) {
            delete static_cast<StatementContext::SELP*>(statement);
        }
        break;
    case S_MAD:
        if (statement) {
            delete static_cast<StatementContext::MAD*>(statement);
        }
        break;
    case S_FMA:
        if (statement) {
            delete static_cast<StatementContext::FMA*>(statement);
        }
        break;
    case S_WMMA:
        if (statement) {
            delete static_cast<StatementContext::WMMA*>(statement);
        }
        break;
    case S_NEG:
        if (statement) {
            delete static_cast<StatementContext::NEG*>(statement);
        }
        break;
    case S_NOT:
        if (statement) {
            delete static_cast<StatementContext::NOT*>(statement);
        }
        break;
    case S_SQRT:
        if (statement) {
            delete static_cast<StatementContext::SQRT*>(statement);
        }
        break;
    case S_COS:
        if (statement) {
            delete static_cast<StatementContext::COS*>(statement);
        }
        break;
    case S_LG2:
        if (statement) {
            delete static_cast<StatementContext::LG2*>(statement);
        }
        break;
    case S_EX2:
        if (statement) {
            delete static_cast<StatementContext::EX2*>(statement);
        }
        break;
    case S_ATOM:
        if (statement) {
            delete static_cast<StatementContext::ATOM*>(statement);
        }
        break;
    case S_XOR:
        if (statement) {
            delete static_cast<StatementContext::XOR*>(statement);
        }
        break;
    case S_ABS:
        if (statement) {
            delete static_cast<StatementContext::ABS*>(statement);
        }
        break;
    case S_SIN:
        if (statement) {
            delete static_cast<StatementContext::SIN*>(statement);
        }
        break;
    case S_RSQRT:
        if (statement) {
            delete static_cast<StatementContext::RSQRT*>(statement);
        }
        break;
    case S_REM:
        if (statement) {
            delete static_cast<StatementContext::REM*>(statement);
        }
        break;
    case S_UNKNOWN:
        // S_UNKNOWN does not need to delete anything
        break;
    }
    statement = nullptr;
}

// Adding StatementContext copy constructor implementation
StatementContext::StatementContext(const StatementContext &other)
    : statementType(other.statementType), statement(nullptr) {
    
    // Deep copy based on statementType
    switch (statementType) {
    case S_REG: {
        if (other.statement) {
            auto source = static_cast<const REG*>(other.statement);
            auto dest = new REG();
            dest->regNum = source->regNum;
            dest->regDataType = source->regDataType;
            dest->regName = source->regName;
            statement = dest;
        }
        break;
    }
    case S_CONST: {
        if (other.statement) {
            auto source = static_cast<const CONST*>(other.statement);
            auto dest = new CONST();
            dest->constAlign = source->constAlign;
            dest->constSize = source->constSize;
            dest->constDataType = source->constDataType;
            dest->constName = source->constName;
            statement = dest;
        }
        break;
    }
    case S_SHARED: {
        if (other.statement) {
            auto source = static_cast<const SHARED*>(other.statement);
            auto dest = new SHARED();
            dest->sharedAlign = source->sharedAlign;
            dest->sharedSize = source->sharedSize;
            dest->sharedDataType = source->sharedDataType;
            dest->sharedName = source->sharedName;
            statement = dest;
        }
        break;
    }
    case S_LOCAL: {
        if (other.statement) {
            auto source = static_cast<const LOCAL*>(other.statement);
            auto dest = new LOCAL();
            dest->localAlign = source->localAlign;
            dest->localSize = source->localSize;
            dest->localDataType = source->localDataType;
            dest->localName = source->localName;
            statement = dest;
        }
        break;
    }
    case S_DOLLOR: {
        if (other.statement) {
            auto source = static_cast<const DOLLOR*>(other.statement);
            auto dest = new DOLLOR();
            dest->dollorName = source->dollorName;
            statement = dest;
        }
        break;
    }
    case S_AT: {
        if (other.statement) {
            auto source = static_cast<const AT*>(other.statement);
            auto dest = new AT();
            // Use OperandContext's copy constructor for deep copy
            dest->atPred = OperandContext(source->atPred);
            dest->atLabelName = source->atLabelName;
            statement = dest;
        }
        break;
    }
    case S_PRAGMA: {
        if (other.statement) {
            auto source = static_cast<const PRAGMA*>(other.statement);
            auto dest = new PRAGMA();
            dest->pragmaString = source->pragmaString;
            statement = dest;
        }
        break;
    }
    case S_RET: {
        // RET has no data members, just create a new instance
        statement = new RET();
        break;
    }
    case S_BAR: {
        if (other.statement) {
            auto source = static_cast<const BAR*>(other.statement);
            auto dest = new BAR();
            dest->braQualifier = source->braQualifier;
            dest->barType = source->barType;
            dest->barId = source->barId;
            statement = dest;
        }
        break;
    }
    case S_BRA: {
        if (other.statement) {
            auto source = static_cast<const BRA*>(other.statement);
            auto dest = new BRA();
            dest->braQualifier = source->braQualifier;
            dest->braTarget = source->braTarget;
            statement = dest;
        }
        break;
    }
    case S_RCP: {
        if (other.statement) {
            auto source = static_cast<const RCP*>(other.statement);
            auto dest = new RCP();
            dest->rcpQualifier = source->rcpQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->rcpOp[0] = OperandContext(source->rcpOp[0]);
            dest->rcpOp[1] = OperandContext(source->rcpOp[1]);
            statement = dest;
        }
        break;
    }
    case S_LD: {
        if (other.statement) {
            auto source = static_cast<const LD*>(other.statement);
            auto dest = new LD();
            dest->ldQualifier = source->ldQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->ldOp[0] = OperandContext(source->ldOp[0]);
            dest->ldOp[1] = OperandContext(source->ldOp[1]);
            statement = dest;
        }
        break;
    }
    case S_MOV: {
        if (other.statement) {
            auto source = static_cast<const MOV*>(other.statement);
            auto dest = new MOV();
            dest->movQualifier = source->movQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->movOp[0] = OperandContext(source->movOp[0]);
            dest->movOp[1] = OperandContext(source->movOp[1]);
            statement = dest;
        }
        break;
    }
    case S_SETP: {
        if (other.statement) {
            auto source = static_cast<const SETP*>(other.statement);
            auto dest = new SETP();
            dest->setpQualifier = source->setpQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->setpOp[0] = OperandContext(source->setpOp[0]);
            dest->setpOp[1] = OperandContext(source->setpOp[1]);
            dest->setpOp[2] = OperandContext(source->setpOp[2]);
            statement = dest;
        }
        break;
    }
    case S_CVTA: {
        if (other.statement) {
            auto source = static_cast<const CVTA*>(other.statement);
            auto dest = new CVTA();
            dest->cvtaQualifier = source->cvtaQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->cvtaOp[0] = OperandContext(source->cvtaOp[0]);
            dest->cvtaOp[1] = OperandContext(source->cvtaOp[1]);
            statement = dest;
        }
        break;
    }
    case S_CVT: {
        if (other.statement) {
            auto source = static_cast<const CVT*>(other.statement);
            auto dest = new CVT();
            dest->cvtQualifier = source->cvtQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->cvtOp[0] = OperandContext(source->cvtOp[0]);
            dest->cvtOp[1] = OperandContext(source->cvtOp[1]);
            statement = dest;
        }
        break;
    }
    case S_MUL: {
        if (other.statement) {
            auto source = static_cast<const MUL*>(other.statement);
            auto dest = new MUL();
            dest->mulQualifier = source->mulQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->mulOp[0] = OperandContext(source->mulOp[0]);
            dest->mulOp[1] = OperandContext(source->mulOp[1]);
            dest->mulOp[2] = OperandContext(source->mulOp[2]);
            statement = dest;
        }
        break;
    }
    case S_DIV: {
        if (other.statement) {
            auto source = static_cast<const DIV*>(other.statement);
            auto dest = new DIV();
            dest->divQualifier = source->divQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->divOp[0] = OperandContext(source->divOp[0]);
            dest->divOp[1] = OperandContext(source->divOp[1]);
            dest->divOp[2] = OperandContext(source->divOp[2]);
            statement = dest;
        }
        break;
    }
    case S_SUB: {
        if (other.statement) {
            auto source = static_cast<const SUB*>(other.statement);
            auto dest = new SUB();
            dest->subQualifier = source->subQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->subOp[0] = OperandContext(source->subOp[0]);
            dest->subOp[1] = OperandContext(source->subOp[1]);
            dest->subOp[2] = OperandContext(source->subOp[2]);
            statement = dest;
        }
        break;
    }
    case S_ADD: {
        if (other.statement) {
            auto source = static_cast<const ADD*>(other.statement);
            auto dest = new ADD();
            dest->addQualifier = source->addQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->addOp[0] = OperandContext(source->addOp[0]);
            dest->addOp[1] = OperandContext(source->addOp[1]);
            dest->addOp[2] = OperandContext(source->addOp[2]);
            statement = dest;
        }
        break;
    }
    case S_SHL: {
        if (other.statement) {
            auto source = static_cast<const SHL*>(other.statement);
            auto dest = new SHL();
            dest->shlQualifier = source->shlQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->shlOp[0] = OperandContext(source->shlOp[0]);
            dest->shlOp[1] = OperandContext(source->shlOp[1]);
            dest->shlOp[2] = OperandContext(source->shlOp[2]);
            statement = dest;
        }
        break;
    }
    case S_SHR: {
        if (other.statement) {
            auto source = static_cast<const SHR*>(other.statement);
            auto dest = new SHR();
            dest->shrQualifier = source->shrQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->shrOp[0] = OperandContext(source->shrOp[0]);
            dest->shrOp[1] = OperandContext(source->shrOp[1]);
            dest->shrOp[2] = OperandContext(source->shrOp[2]);
            statement = dest;
        }
        break;
    }
    case S_MAX: {
        if (other.statement) {
            auto source = static_cast<const MAX*>(other.statement);
            auto dest = new MAX();
            dest->maxQualifier = source->maxQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->maxOp[0] = OperandContext(source->maxOp[0]);
            dest->maxOp[1] = OperandContext(source->maxOp[1]);
            dest->maxOp[2] = OperandContext(source->maxOp[2]);
            statement = dest;
        }
        break;
    }
    case S_MIN: {
        if (other.statement) {
            auto source = static_cast<const MIN*>(other.statement);
            auto dest = new MIN();
            dest->minQualifier = source->minQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->minOp[0] = OperandContext(source->minOp[0]);
            dest->minOp[1] = OperandContext(source->minOp[1]);
            dest->minOp[2] = OperandContext(source->minOp[2]);
            statement = dest;
        }
        break;
    }
    case S_AND: {
        if (other.statement) {
            auto source = static_cast<const AND*>(other.statement);
            auto dest = new AND();
            dest->andQualifier = source->andQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->andOp[0] = OperandContext(source->andOp[0]);
            dest->andOp[1] = OperandContext(source->andOp[1]);
            dest->andOp[2] = OperandContext(source->andOp[2]);
            statement = dest;
        }
        break;
    }
    case S_OR: {
        if (other.statement) {
            auto source = static_cast<const OR*>(other.statement);
            auto dest = new OR();
            dest->orQualifier = source->orQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->orOp[0] = OperandContext(source->orOp[0]);
            dest->orOp[1] = OperandContext(source->orOp[1]);
            dest->orOp[2] = OperandContext(source->orOp[2]);
            statement = dest;
        }
        break;
    }
    case S_ST: {
        if (other.statement) {
            auto source = static_cast<const ST*>(other.statement);
            auto dest = new ST();
            dest->stQualifier = source->stQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->stOp[0] = OperandContext(source->stOp[0]);
            dest->stOp[1] = OperandContext(source->stOp[1]);
            statement = dest;
        }
        break;
    }
    case S_SELP: {
        if (other.statement) {
            auto source = static_cast<const SELP*>(other.statement);
            auto dest = new SELP();
            dest->selpQualifier = source->selpQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->selpOp[0] = OperandContext(source->selpOp[0]);
            dest->selpOp[1] = OperandContext(source->selpOp[1]);
            dest->selpOp[2] = OperandContext(source->selpOp[2]);
            dest->selpOp[3] = OperandContext(source->selpOp[3]);
            statement = dest;
        }
        break;
    }
    case S_MAD: {
        if (other.statement) {
            auto source = static_cast<const MAD*>(other.statement);
            auto dest = new MAD();
            dest->madQualifier = source->madQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->madOp[0] = OperandContext(source->madOp[0]);
            dest->madOp[1] = OperandContext(source->madOp[1]);
            dest->madOp[2] = OperandContext(source->madOp[2]);
            dest->madOp[3] = OperandContext(source->madOp[3]);
            statement = dest;
        }
        break;
    }
    case S_FMA: {
        if (other.statement) {
            auto source = static_cast<const FMA*>(other.statement);
            auto dest = new FMA();
            dest->fmaQualifier = source->fmaQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->fmaOp[0] = OperandContext(source->fmaOp[0]);
            dest->fmaOp[1] = OperandContext(source->fmaOp[1]);
            dest->fmaOp[2] = OperandContext(source->fmaOp[2]);
            dest->fmaOp[3] = OperandContext(source->fmaOp[3]);
            statement = dest;
        }
        break;
    }
    case S_WMMA: {
        if (other.statement) {
            auto source = static_cast<const WMMA*>(other.statement);
            auto dest = new WMMA();
            dest->wmmaType = source->wmmaType;
            dest->wmmaQualifier = source->wmmaQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->wmmaOp[0] = OperandContext(source->wmmaOp[0]);
            dest->wmmaOp[1] = OperandContext(source->wmmaOp[1]);
            dest->wmmaOp[2] = OperandContext(source->wmmaOp[2]);
            dest->wmmaOp[3] = OperandContext(source->wmmaOp[3]);
            statement = dest;
        }
        break;
    }
    case S_NEG: {
        if (other.statement) {
            auto source = static_cast<const NEG*>(other.statement);
            auto dest = new NEG();
            dest->negQualifier = source->negQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->negOp[0] = OperandContext(source->negOp[0]);
            dest->negOp[1] = OperandContext(source->negOp[1]);
            statement = dest;
        }
        break;
    }
    case S_NOT: {
        if (other.statement) {
            auto source = static_cast<const NOT*>(other.statement);
            auto dest = new NOT();
            dest->notQualifier = source->notQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->notOp[0] = OperandContext(source->notOp[0]);
            dest->notOp[1] = OperandContext(source->notOp[1]);
            statement = dest;
        }
        break;
    }
    case S_SQRT: {
        if (other.statement) {
            auto source = static_cast<const SQRT*>(other.statement);
            auto dest = new SQRT();
            dest->sqrtQualifier = source->sqrtQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->sqrtOp[0] = OperandContext(source->sqrtOp[0]);
            dest->sqrtOp[1] = OperandContext(source->sqrtOp[1]);
            statement = dest;
        }
        break;
    }
    case S_COS: {
        if (other.statement) {
            auto source = static_cast<const COS*>(other.statement);
            auto dest = new COS();
            dest->cosQualifier = source->cosQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->cosOp[0] = OperandContext(source->cosOp[0]);
            dest->cosOp[1] = OperandContext(source->cosOp[1]);
            statement = dest;
        }
        break;
    }
    case S_LG2: {
        if (other.statement) {
            auto source = static_cast<const LG2*>(other.statement);
            auto dest = new LG2();
            dest->lg2Qualifier = source->lg2Qualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->lg2Op[0] = OperandContext(source->lg2Op[0]);
            dest->lg2Op[1] = OperandContext(source->lg2Op[1]);
            statement = dest;
        }
        break;
    }
    case S_EX2: {
        if (other.statement) {
            auto source = static_cast<const EX2*>(other.statement);
            auto dest = new EX2();
            dest->ex2Qualifier = source->ex2Qualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->ex2Op[0] = OperandContext(source->ex2Op[0]);
            dest->ex2Op[1] = OperandContext(source->ex2Op[1]);
            statement = dest;
        }
        break;
    }
    case S_ATOM: {
        if (other.statement) {
            auto source = static_cast<const ATOM*>(other.statement);
            auto dest = new ATOM();
            dest->atomQualifier = source->atomQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->atomOp[0] = OperandContext(source->atomOp[0]);
            dest->atomOp[1] = OperandContext(source->atomOp[1]);
            dest->atomOp[2] = OperandContext(source->atomOp[2]);
            dest->atomOp[3] = OperandContext(source->atomOp[3]);
            dest->operandNum = source->operandNum;
            statement = dest;
        }
        break;
    }
    case S_XOR: {
        if (other.statement) {
            auto source = static_cast<const XOR*>(other.statement);
            auto dest = new XOR();
            dest->xorQualifier = source->xorQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->xorOp[0] = OperandContext(source->xorOp[0]);
            dest->xorOp[1] = OperandContext(source->xorOp[1]);
            dest->xorOp[2] = OperandContext(source->xorOp[2]);
            statement = dest;
        }
        break;
    }
    case S_ABS: {
        if (other.statement) {
            auto source = static_cast<const ABS*>(other.statement);
            auto dest = new ABS();
            dest->absQualifier = source->absQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->absOp[0] = OperandContext(source->absOp[0]);
            dest->absOp[1] = OperandContext(source->absOp[1]);
            statement = dest;
        }
        break;
    }
    case S_SIN: {
        if (other.statement) {
            auto source = static_cast<const SIN*>(other.statement);
            auto dest = new SIN();
            dest->sinQualifier = source->sinQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->sinOp[0] = OperandContext(source->sinOp[0]);
            dest->sinOp[1] = OperandContext(source->sinOp[1]);
            statement = dest;
        }
        break;
    }
    case S_RSQRT: {
        if (other.statement) {
            auto source = static_cast<const RSQRT*>(other.statement);
            auto dest = new RSQRT();
            dest->rsqrtQualifier = source->rsqrtQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->rsqrtOp[0] = OperandContext(source->rsqrtOp[0]);
            dest->rsqrtOp[1] = OperandContext(source->rsqrtOp[1]);
            statement = dest;
        }
        break;
    }
    case S_REM: {
        if (other.statement) {
            auto source = static_cast<const REM*>(other.statement);
            auto dest = new REM();
            dest->remQualifier = source->remQualifier;
            // Use OperandContext's copy constructor for deep copy
            dest->remOp[0] = OperandContext(source->remOp[0]);
            dest->remOp[1] = OperandContext(source->remOp[1]);
            dest->remOp[2] = OperandContext(source->remOp[2]);
            statement = dest;
        }
        break;
    }

    case S_UNKNOWN: {
        // S_UNKNOWN does not need to copy anything
        statement = nullptr;
        break;
    }
    }
}
