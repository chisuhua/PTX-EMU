#include "ptxsim/thread_context.h"
#include "../utils/qualifier_utils.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/interpreter.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifdef DEBUGINTE
extern bool sync_thread;
#endif
#ifdef LOGINTE
extern bool IFLOG();
#endif

void ThreadContext::init(
    dim3 &blockIdx, dim3 &threadIdx, dim3 GridDim, dim3 BlockDim,
    std::vector<StatementContext> &statements,
    std::map<std::string, PtxInterpreter::Symtable *> &name2Share,
    std::map<std::string, PtxInterpreter::Symtable *> &name2Sym,
    std::map<std::string, int> &label2pc) {
    this->BlockIdx = blockIdx;
    this->ThreadIdx = threadIdx;
    this->GridDim = GridDim;
    this->BlockDim = BlockDim;
    this->statements = &statements;
    this->name2Share = &name2Share;
    this->name2Sym = name2Sym;
    this->label2pc = label2pc;
    this->pc = 0;
    this->state = RUN;
}

void ThreadContext::clearIMM_VEC() {
    while (!imm.empty()) {
        delete imm.front();
        imm.pop();
    }
    while (!vec.empty()) {
        for (auto e : vec.front()->vec)
            delete (char *)e;
        delete vec.front();
        vec.pop();
    }
}

EXE_STATE ThreadContext::exe_once() {
    if (state == EXIT || state == BAR)
        return state;
    _exe_once();
    return state;
}

void ThreadContext::_exe_once() {
    assert(state == RUN);
    assert(pc >= 0 && pc < statements->size());

    StatementContext &statement = (*statements)[pc];
    handleStatement(statement);

    pc++;
    if (pc >= statements->size())
        state = EXIT;

    clearIMM_VEC();
}

void ThreadContext::handleStatement(StatementContext &statement) {
    switch (statement.statementType) {
    case S_REG:
        handle_reg((StatementContext::REG *)statement.statement);
        break;
    case S_SHARED:
        handle_shared((StatementContext::SHARED *)statement.statement);
        break;
    case S_LOCAL:
        handle_local((StatementContext::LOCAL *)statement.statement);
        break;
    case S_DOLLOR:
        handle_dollor((StatementContext::DOLLOR *)statement.statement);
        break;
    case S_AT:
        handle_at((StatementContext::AT *)statement.statement);
        break;
    case S_PRAGMA:
        handle_pragma((StatementContext::PRAGMA *)statement.statement);
        break;
    case S_RET:
        handle_ret((StatementContext::RET *)statement.statement);
        break;
    case S_BAR:
        handle_bar((StatementContext::BAR *)statement.statement);
        break;
    case S_BRA:
        handle_bra((StatementContext::BRA *)statement.statement);
        break;
    case S_RCP:
        handle_rcp((StatementContext::RCP *)statement.statement);
        break;
    case S_LD:
        handle_ld((StatementContext::LD *)statement.statement);
        break;
    case S_MOV:
        handle_mov((StatementContext::MOV *)statement.statement);
        break;
    case S_SETP:
        handle_setp((StatementContext::SETP *)statement.statement);
        break;
    case S_CVTA:
        handle_cvta((StatementContext::CVTA *)statement.statement);
        break;
    case S_CVT:
        handle_cvt((StatementContext::CVT *)statement.statement);
        break;
    case S_MUL:
        handle_mul((StatementContext::MUL *)statement.statement);
        break;
    case S_DIV:
        handle_div((StatementContext::DIV *)statement.statement);
        break;
    case S_SUB:
        handle_sub((StatementContext::SUB *)statement.statement);
        break;
    case S_ADD:
        handle_add((StatementContext::ADD *)statement.statement);
        break;
    case S_SHL:
        handle_shl((StatementContext::SHL *)statement.statement);
        break;
    case S_SHR:
        handle_shr((StatementContext::SHR *)statement.statement);
        break;
    case S_MAX:
        handle_max((StatementContext::MAX *)statement.statement);
        break;
    case S_MIN:
        handle_min((StatementContext::MIN *)statement.statement);
        break;
    case S_AND:
        handle_and((StatementContext::AND *)statement.statement);
        break;
    case S_OR:
        handle_or((StatementContext::OR *)statement.statement);
        break;
    case S_ST:
        handle_st((StatementContext::ST *)statement.statement);
        break;
    case S_SELP:
        handle_selp((StatementContext::SELP *)statement.statement);
        break;
    case S_MAD:
        handle_mad((StatementContext::MAD *)statement.statement);
        break;
    case S_FMA:
        handle_fma((StatementContext::FMA *)statement.statement);
        break;
    case S_NEG:
        handle_neg((StatementContext::NEG *)statement.statement);
        break;
    case S_NOT:
        handle_not((StatementContext::NOT *)statement.statement);
        break;
    case S_SQRT:
        handle_sqrt((StatementContext::SQRT *)statement.statement);
        break;
    case S_COS:
        handle_cos((StatementContext::COS *)statement.statement);
        break;
    case S_LG2:
        handle_lg2((StatementContext::LG2 *)statement.statement);
        break;
    case S_EX2:
        handle_ex2((StatementContext::EX2 *)statement.statement);
        break;
    case S_ATOM:
        handle_atom((StatementContext::ATOM *)statement.statement);
        break;
    case S_XOR:
        handle_xor((StatementContext::XOR *)statement.statement);
        break;
    case S_ABS:
        handle_abs((StatementContext::ABS *)statement.statement);
        break;
    case S_SIN:
        handle_sin((StatementContext::SIN *)statement.statement);
        break;
    case S_REM:
        handle_rem((StatementContext::REM *)statement.statement);
        break;
    case S_RSQRT:
        handle_rsqrt((StatementContext::RSQRT *)statement.statement);
        break;
    case S_WMMA:
        handle_wmma((StatementContext::WMMA *)statement.statement);
        break;
    case S_CONST:
        // Constants are already set up before execution starts
        break;
    default:
        assert(false && "Unsupported statement type");
    }
}

void ThreadContext::handle_reg(StatementContext::REG *ss) {
    for (int i = 0; i < ss->regName.size(); i++) {
        PtxInterpreter::Reg *r = new PtxInterpreter::Reg();
        r->byteNum = Q2bytes(ss->regDataType.back());
        r->elementNum = 1;
        r->name = ss->regName[i];
        r->regType = ss->regDataType.back();
        r->addr = new char[r->byteNum];
        memset(r->addr, 0, r->byteNum);
        name2Reg[r->name] = r;
    }
}

void ThreadContext::handle_shared(StatementContext::SHARED *ss) {
    PtxInterpreter::Symtable *s = new PtxInterpreter::Symtable();
    s->byteNum = Q2bytes(ss->sharedDataType.back()) * ss->sharedSize;
    s->elementNum = ss->sharedSize;
    s->name = ss->sharedName;
    s->symType = ss->sharedDataType.back();
    s->val = (uint64_t) new char[s->byteNum];
    memset((void *)s->val, 0, s->byteNum);
    (*name2Share)[s->name] = s;
    name2Sym[s->name] = s;
}

void ThreadContext::handle_local(StatementContext::LOCAL *ss) {
    PtxInterpreter::Symtable *s = new PtxInterpreter::Symtable();
    s->byteNum = Q2bytes(ss->localDataType.back()) * ss->localSize;
    s->elementNum = ss->localSize;
    s->name = ss->localName;
    s->symType = ss->localDataType.back();
    s->val = (uint64_t) new char[s->byteNum];
    memset((void *)s->val, 0, s->byteNum);
    name2Sym[s->name] = s;
}

void ThreadContext::handle_dollor(StatementContext::DOLLOR *ss) {
    // Labels are already set up before execution starts
    // Nothing to do here during execution
}

bool ThreadContext::QvecHasQ(std::vector<Qualifier> &qvec, Qualifier q) {
    return std::find(qvec.begin(), qvec.end(), q) != qvec.end();
}

Qualifier ThreadContext::getDataType(std::vector<Qualifier> &q) {
    for (auto e : q) {
        switch (e) {
        case Qualifier::Q_U64:
        case Qualifier::Q_S64:
        case Qualifier::Q_B64:
        case Qualifier::Q_F64:
        case Qualifier::Q_U32:
        case Qualifier::Q_S32:
        case Qualifier::Q_B32:
        case Qualifier::Q_F32:
        case Qualifier::Q_U16:
        case Qualifier::Q_S16:
        case Qualifier::Q_B16:
        case Qualifier::Q_F16:
        case Qualifier::Q_U8:
        case Qualifier::Q_S8:
        case Qualifier::Q_B8:
        case Qualifier::Q_F8:
        case Qualifier::Q_PRED:
            return e;
        default:
            continue;
        }
    }
    return Qualifier::Q_B32; // Default
}

int ThreadContext::getBytes(std::vector<Qualifier> &q) {
    Qualifier dt = getDataType(q);
    return Q2bytes(dt);
}

int ThreadContext::getBytes(Qualifier q) { return Q2bytes(q); }

void ThreadContext::setIMM(std::string s, Qualifier q) {
    PtxInterpreter::IMM *imm_val = new PtxInterpreter::IMM();
    imm_val->type = q;

    switch (q) {
    case Qualifier::Q_U8:
        imm_val->data.u8 = std::stoul(s);
        break;
    case Qualifier::Q_U16:
        imm_val->data.u16 = std::stoul(s);
        break;
    case Qualifier::Q_U32:
        imm_val->data.u32 = std::stoul(s);
        break;
    case Qualifier::Q_U64:
        imm_val->data.u64 = std::stoull(s);
        break;
    case Qualifier::Q_S8:
        imm_val->data.u8 = std::stoi(s);
        break;
    case Qualifier::Q_S16:
        imm_val->data.u16 = std::stoi(s);
        break;
    case Qualifier::Q_S32:
        imm_val->data.u32 = std::stoi(s);
        break;
    case Qualifier::Q_S64:
        imm_val->data.u64 = std::stoll(s);
        break;
    case Qualifier::Q_F32:
        imm_val->data.f32 = std::stof(s);
        break;
    case Qualifier::Q_F64:
        imm_val->data.f64 = std::stod(s);
        break;
    default:
        assert(false && "Unsupported immediate type");
    }

    imm.push(imm_val);
}

bool ThreadContext::isIMMorVEC(OperandContext &op) {
    return op.operandType == O_IMM || op.operandType == O_VEC;
}

void *ThreadContext::getOperandAddr(OperandContext &op,
                                    std::vector<Qualifier> &q) {
    if (op.operandType == O_REG) {
        return getRegAddr((OperandContext::REG *)op.operand);
    } else if (op.operandType == O_FA) {
        return getFaAddr((OperandContext::FA *)op.operand, q);
    } else if (op.operandType == O_IMM) {
        auto imm_op = (OperandContext::IMM *)op.operand;
        setIMM(imm_op->immVal, getDataType(q));
        return imm.back();
    } else if (op.operandType == O_PRED) {
        auto pred_op = (OperandContext::PRED *)op.operand;
        if (pred_op->isNot) {
            // Handle negated predicate
            assert(false && "Negated predicates not implemented yet");
        }
        return getRegAddr((OperandContext::REG *)pred_op->pred->operand);
    } else if (op.operandType == O_VEC) {
        auto vec_op = (OperandContext::VEC *)op.operand;
        PtxInterpreter::VEC *vec_val = new PtxInterpreter::VEC();
        for (auto &e : vec_op->vec) {
            vec_val->vec.push_back(getOperandAddr(e, q));
        }
        vec.push(vec_val);
        return vec_val;
    }

    assert(false && "Unsupported operand type");
    return nullptr;
}

void *ThreadContext::getRegAddr(OperandContext::REG *regContext) {
    auto iter = name2Reg.find(regContext->regName);
    if (iter != name2Reg.end()) {
        return iter->second->addr;
    }

    // Check if it's a special register
    if (regContext->regName == "%ctaid.x")
        return &BlockIdx.x;
    if (regContext->regName == "%ctaid.y")
        return &BlockIdx.y;
    if (regContext->regName == "%ctaid.z")
        return &BlockIdx.z;
    if (regContext->regName == "%tid.x")
        return &ThreadIdx.x;
    if (regContext->regName == "%tid.y")
        return &ThreadIdx.y;
    if (regContext->regName == "%tid.z")
        return &ThreadIdx.z;
    if (regContext->regName == "%nctaid.x")
        return &GridDim.x;
    if (regContext->regName == "%nctaid.y")
        return &GridDim.y;
    if (regContext->regName == "%nctaid.z")
        return &GridDim.z;
    if (regContext->regName == "%ntid.x")
        return &BlockDim.x;
    if (regContext->regName == "%ntid.y")
        return &BlockDim.y;
    if (regContext->regName == "%ntid.z")
        return &BlockDim.z;

    assert(false && "Register not found");
    return nullptr;
}

void *ThreadContext::getFaAddr(OperandContext::FA *fa,
                               std::vector<Qualifier> &q) {
    uint64_t base = 0;
    int offsetByte = 0;

    // Get base address
    if (fa->baseType == OperandContext::FA::CONSTANT) {
        auto sym_iter = name2Sym.find(fa->baseName);
        if (sym_iter != name2Sym.end()) {
            base = sym_iter->second->val;
        } else {
            assert(false && "Symbol not found");
        }
    } else if (fa->baseType == OperandContext::FA::SHARED) {
        auto sym_iter = name2Share->find(fa->baseName);
        if (sym_iter != name2Share->end()) {
            base = sym_iter->second->val;
        } else {
            assert(false && "Shared memory symbol not found");
        }
    } else {
        assert(false && "Unsupported base type");
    }

    // Add offset
    if (fa->offsetType == OperandContext::FA::IMMEDIATE) {
        offsetByte = std::stoi(fa->offsetVal);
    } else if (fa->offsetType == OperandContext::FA::REGISTER) {
        OperandContext tmp;
        tmp.operandType = O_REG;
        tmp.operand = new OperandContext::REG{fa->offsetVal};
        void *offsetAddr = getOperandAddr(tmp, q);
        memcpy(&offsetByte, offsetAddr, sizeof(int));
    }

    // Calculate element size
    int elementSize = getBytes(q);
    return (void *)(base + offsetByte * elementSize);
}

void ThreadContext::mov(void *from, void *to, std::vector<Qualifier> &q) {
    int bytes = getBytes(q);
    int elements = QvecHasQ(q, Qualifier::Q_V2)   ? 2
                   : QvecHasQ(q, Qualifier::Q_V4) ? 4
                                                  : 1;

    if (from == nullptr) {
        memset(to, 0, bytes * elements);
        return;
    }

    if (QvecHasQ(q, Qualifier::Q_HI)) {
        uint64_t val;
        memcpy(&val, from, 8);
        uint32_t hi = (uint32_t)(val >> 32);
        memcpy(to, &hi, 4);
    } else if (QvecHasQ(q, Qualifier::Q_LO)) {
        uint64_t val;
        memcpy(&val, from, 8);
        uint32_t lo = (uint32_t)(val & 0xFFFFFFFF);
        memcpy(to, &lo, 4);
    } else {
        memcpy(to, from, bytes * elements);
    }
}

void ThreadContext::handle_ret(StatementContext::RET *ss) { state = EXIT; }

void ThreadContext::handle_bar(StatementContext::BAR *ss) {
    if (ss->barType == "sync") {
        state = BAR;
#ifdef DEBUGINTE
        sync_thread = 1;
#endif
#ifdef LOGINTE
        if (IFLOG()) {
            printf("INTE: Thread(%d,%d,%d) in Block(%d,%d,%d) bar.sync\n",
                   ThreadIdx.x, ThreadIdx.y, ThreadIdx.z, BlockIdx.x,
                   BlockIdx.y, BlockIdx.z);
        }
#endif
    } else {
        assert(false && "Unsupported barrier type");
    }
}

void ThreadContext::handle_bra(StatementContext::BRA *ss) {
    auto iter = label2pc.find(ss->braTarget);
    assert(iter != label2pc.end());
    pc = iter->second -
         1; // -1 because pc will be incremented after this instruction
}

// Basic implementation for other handlers - they'll be fleshed out as needed
void ThreadContext::handle_at(
    StatementContext::AT *ss) { /* Not implemented yet */ }
void ThreadContext::handle_pragma(
    StatementContext::PRAGMA *ss) { /* Not implemented yet */ }
void ThreadContext::handle_rcp(
    StatementContext::RCP *ss) { /* Not implemented yet */ }
void ThreadContext::handle_ld(
    StatementContext::LD *ss) { /* Not implemented yet */ }
void ThreadContext::handle_mov(
    StatementContext::MOV *ss) { /* Not implemented yet */ }
void ThreadContext::handle_setp(
    StatementContext::SETP *ss) { /* Not implemented yet */ }
void ThreadContext::handle_cvta(
    StatementContext::CVTA *ss) { /* Not implemented yet */ }
void ThreadContext::handle_cvt(
    StatementContext::CVT *ss) { /* Not implemented yet */ }
void ThreadContext::handle_mul(
    StatementContext::MUL *ss) { /* Not implemented yet */ }
void ThreadContext::handle_div(
    StatementContext::DIV *ss) { /* Not implemented yet */ }
void ThreadContext::handle_sub(
    StatementContext::SUB *ss) { /* Not implemented yet */ }
void ThreadContext::handle_add(
    StatementContext::ADD *ss) { /* Not implemented yet */ }
void ThreadContext::handle_shl(
    StatementContext::SHL *ss) { /* Not implemented yet */ }
void ThreadContext::handle_shr(
    StatementContext::SHR *ss) { /* Not implemented yet */ }
void ThreadContext::handle_max(
    StatementContext::MAX *ss) { /* Not implemented yet */ }
void ThreadContext::handle_min(
    StatementContext::MIN *ss) { /* Not implemented yet */ }
void ThreadContext::handle_and(
    StatementContext::AND *ss) { /* Not implemented yet */ }
void ThreadContext::handle_or(
    StatementContext::OR *ss) { /* Not implemented yet */ }
void ThreadContext::handle_st(
    StatementContext::ST *ss) { /* Not implemented yet */ }
void ThreadContext::handle_selp(
    StatementContext::SELP *ss) { /* Not implemented yet */ }
void ThreadContext::handle_mad(
    StatementContext::MAD *ss) { /* Not implemented yet */ }
void ThreadContext::handle_fma(
    StatementContext::FMA *ss) { /* Not implemented yet */ }
void ThreadContext::handle_neg(
    StatementContext::NEG *ss) { /* Not implemented yet */ }
void ThreadContext::handle_not(
    StatementContext::NOT *ss) { /* Not implemented yet */ }
void ThreadContext::handle_sqrt(
    StatementContext::SQRT *ss) { /* Not implemented yet */ }
void ThreadContext::handle_cos(
    StatementContext::COS *ss) { /* Not implemented yet */ }
void ThreadContext::handle_lg2(
    StatementContext::LG2 *ss) { /* Not implemented yet */ }
void ThreadContext::handle_ex2(
    StatementContext::EX2 *ss) { /* Not implemented yet */ }
void ThreadContext::handle_atom(
    StatementContext::ATOM *ss) { /* Not implemented yet */ }
void ThreadContext::handle_xor(
    StatementContext::XOR *ss) { /* Not implemented yet */ }
void ThreadContext::handle_abs(
    StatementContext::ABS *ss) { /* Not implemented yet */ }
void ThreadContext::handle_sin(
    StatementContext::SIN *ss) { /* Not implemented yet */ }
void ThreadContext::handle_rem(
    StatementContext::REM *ss) { /* Not implemented yet */ }
void ThreadContext::handle_rsqrt(
    StatementContext::RSQRT *ss) { /* Not implemented yet */ }
void ThreadContext::handle_wmma(
    StatementContext::WMMA *ss) { /* Not implemented yet */ }