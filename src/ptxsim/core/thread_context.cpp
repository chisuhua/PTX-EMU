#include "ptxsim/thread_context.h"
#include "../utils/qualifier_utils.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/interpreter.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>


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
    for (int i = 0; i < ss->regNum; i++) {
        PtxInterpreter::Reg *r = new PtxInterpreter::Reg();
        r->byteNum = Q2bytes(ss->regDataType.back());
        r->elementNum = 1;
        // 存储完整的寄存器名称，包括索引部分
        r->name = ss->regName + std::to_string(i);
        r->regType = ss->regDataType.back();
        r->addr = new char[r->byteNum];
        memset(r->addr, 0, r->byteNum);
        name2Reg[r->name] = r;
        std::cout << "Registered register: " << r->name << std::endl;
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
    // Check if it's a special register (before combining names)
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

    // 首先尝试直接按regName查找
    auto iter = name2Reg.find(regContext->regName);
    if (iter != name2Reg.end()) {
        return iter->second->addr;
    }

    // 如果没找到，尝试组合名称查找（例如 regName="r", regIdx=1 组合成 "r1"）
    std::string combinedName =
        regContext->regName + std::to_string(regContext->regIdx);
    iter = name2Reg.find(combinedName);
    if (iter != name2Reg.end()) {
        return iter->second->addr;
    }

    // 对于特殊寄存器，不应该创建临时寄存器，而应该报错
    if (regContext->regName[0] == '%') {
        std::cerr << "Error: Special register '" << regContext->regName 
                  << "' not found" << std::endl;
        assert(false && "Special register not found");
        return nullptr;
    }
    
    // 创建一个临时的普通寄存器
    PtxInterpreter::Reg *r = new PtxInterpreter::Reg();
    r->byteNum = 4; // 默认4字节
    r->elementNum = 1;
    r->name = combinedName; // 使用组合名称
    r->regType = Qualifier::Q_B32;
    r->addr = new char[r->byteNum];
    memset(r->addr, 0, r->byteNum);
    name2Reg[r->name] = r;
    return r->addr;
}

void *ThreadContext::getFaAddr(OperandContext::FA *fa,
                               std::vector<Qualifier> &q) {
    uint64_t base = 0;
    int offsetByte = 0;

    // Get base address
    if (fa->baseType == OperandContext::FA::CONSTANT) {
        // 检查符号名是否为空
        if (fa->baseName.empty()) {
            std::cerr << "Warning: Empty constant symbol name" << std::endl;
            // 返回一个默认地址
            static char dummy_addr[8] = {0};
            return dummy_addr;
        }
        
        auto sym_iter = name2Sym.find(fa->baseName);
        if (sym_iter != name2Sym.end()) {
            base = sym_iter->second->val;
        } else {
            // 输出调试信息而不是直接断言失败
            std::cerr << "Warning: Constant symbol '" << fa->baseName << "' not found" << std::endl;
            // 创建一个临时的符号表项
            PtxInterpreter::Symtable *s = new PtxInterpreter::Symtable();
            s->symType = Qualifier::Q_B32;
            s->byteNum = 4;
            s->elementNum = 1;
            s->name = fa->baseName;
            s->val = 0; // 默认值
            name2Sym[fa->baseName] = s;
            base = s->val;
        }
    } else if (fa->baseType == OperandContext::FA::SHARED) {
        auto sym_iter = name2Share->find(fa->baseName);
        if (sym_iter != name2Share->end()) {
            base = sym_iter->second->val;
        } else {
            // 输出调试信息而不是直接断言失败
            std::cerr << "Warning: Shared memory symbol '" << fa->baseName << "' not found" << std::endl;
            // 创建一个临时的符号表项
            PtxInterpreter::Symtable *s = new PtxInterpreter::Symtable();
            s->symType = Qualifier::Q_B32;
            s->byteNum = 4;
            s->elementNum = 1;
            s->name = fa->baseName;
            s->val = 0; // 默认值
            (*name2Share)[fa->baseName] = s;
            base = s->val;
        }
    } else {
        std::cerr << "Error: Unsupported base type in FA operand" << std::endl;
        assert(false && "Unsupported base type");
    }

    // Add offset
    if (fa->offsetType == OperandContext::FA::IMMEDIATE) {
        offsetByte = std::stoi(fa->offsetVal);
    } else if (fa->offsetType == OperandContext::FA::REGISTER) {
        OperandContext tmp;
        tmp.operandType = O_REG;
        OperandContext::REG* regOperand = new OperandContext::REG();
        regOperand->regName = fa->offsetVal;
        regOperand->regIdx = 0;
        tmp.operand = regOperand;
        void *offsetAddr = getOperandAddr(tmp, q);
        memcpy(&offsetByte, offsetAddr, sizeof(int));
        delete regOperand; // 清理临时分配的内存
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

void ThreadContext::handle_setp(StatementContext::SETP *ss) {
    // op0
    void *to = getOperandAddr(ss->setpOp[0], ss->setpQualifier);

    // op1
    void *op1 = getOperandAddr(ss->setpOp[1], ss->setpQualifier);

    // op2
    void *op2 = getOperandAddr(ss->setpOp[2], ss->setpQualifier);

    // get compare op
    Qualifier cmpOp = getCMPOP(ss->setpQualifier);

    // exe setp
    setp(to, op1, op2, cmpOp, ss->setpQualifier);
}

void ThreadContext::handle_cvt(StatementContext::CVT *ss) {
    // op0
    void *to = getOperandAddr(ss->cvtOp[0], ss->cvtQualifier);

    // op1
    void *from = getOperandAddr(ss->cvtOp[1], ss->cvtQualifier);

    // exe cvt
    cvt(to, from, ss->cvtQualifier, ss->cvtQualifier);
}

void ThreadContext::handle_rcp(StatementContext::RCP *ss) {
    // op0
    void *to = getOperandAddr(ss->rcpOp[0], ss->rcpQualifier);

    // op1
    void *op = getOperandAddr(ss->rcpOp[1], ss->rcpQualifier);

    // exe rcp
    rcp(to, op, ss->rcpQualifier);
}

void ThreadContext::handle_atom(StatementContext::ATOM *ss) {
    /* Not implemented yet */
}

void ThreadContext::handle_neg(StatementContext::NEG *ss) {
    // op0
    void *to = getOperandAddr(ss->negOp[0], ss->negQualifier);

    // op1
    void *op = getOperandAddr(ss->negOp[1], ss->negQualifier);

    // exe neg
    neg(to, op, ss->negQualifier);
}

void ThreadContext::handle_sqrt(StatementContext::SQRT *ss) {
    // op0
    void *to = getOperandAddr(ss->sqrtOp[0], ss->sqrtQualifier);

    // op1
    void *op = getOperandAddr(ss->sqrtOp[1], ss->sqrtQualifier);

    // exe sqrt
    sqrt(to, op, ss->sqrtQualifier);
}

void ThreadContext::handle_abs(StatementContext::ABS *ss) {
    // op0
    void *to = getOperandAddr(ss->absOp[0], ss->absQualifier);

    // op1
    void *op = getOperandAddr(ss->absOp[1], ss->absQualifier);

    // exe abs
    abs(to, op, ss->absQualifier);
}

void ThreadContext::handle_min(StatementContext::MIN *ss) {
    // op0
    void *to = getOperandAddr(ss->minOp[0], ss->minQualifier);

    // op1
    void *op1 = getOperandAddr(ss->minOp[1], ss->minQualifier);

    // op2
    void *op2 = getOperandAddr(ss->minOp[2], ss->minQualifier);

    // exe min
    min(to, op1, op2, ss->minQualifier);
}

void ThreadContext::handle_max(StatementContext::MAX *ss) {
    // op0
    void *to = getOperandAddr(ss->maxOp[0], ss->maxQualifier);

    // op1
    void *op1 = getOperandAddr(ss->maxOp[1], ss->maxQualifier);

    // op2
    void *op2 = getOperandAddr(ss->maxOp[2], ss->maxQualifier);

    // exe max
    max(to, op1, op2, ss->maxQualifier);
}

void ThreadContext::handle_mad(StatementContext::MAD *ss) {
    // op0
    void *to = getOperandAddr(ss->madOp[0], ss->madQualifier);

    // op1
    void *op1 = getOperandAddr(ss->madOp[1], ss->madQualifier);

    // op2
    void *op2 = getOperandAddr(ss->madOp[2], ss->madQualifier);

    // op3
    void *op3 = getOperandAddr(ss->madOp[3], ss->madQualifier);

    // exe mad
    mad(to, op1, op2, op3, ss->madQualifier);
}

void ThreadContext::handle_fma(StatementContext::FMA *ss) {
    // op0
    void *to = getOperandAddr(ss->fmaOp[0], ss->fmaQualifier);

    // op1
    void *op1 = getOperandAddr(ss->fmaOp[1], ss->fmaQualifier);

    // op2
    void *op2 = getOperandAddr(ss->fmaOp[2], ss->fmaQualifier);

    // op3
    void *op3 = getOperandAddr(ss->fmaOp[3], ss->fmaQualifier);

    // exe fma
    fma(to, op1, op2, op3, ss->fmaQualifier);
}

// Basic implementation for other handlers - they'll be fleshed out as needed

void ThreadContext::handle_at(
    StatementContext::AT *ss) { /* Not implemented yet */ }
void ThreadContext::handle_pragma(
    StatementContext::PRAGMA *ss) { /* Not implemented yet */ }
void ThreadContext::handle_ld(StatementContext::LD *ss) {
    // process op0
    void *to = getOperandAddr(ss->ldOp[0], ss->ldQualifier);

    // process op1
    void *from = getOperandAddr(ss->ldOp[1], ss->ldQualifier);

    // exe ld
    if (QvecHasQ(ss->ldQualifier, Qualifier::Q_V2)) {
        uint64_t step = getBytes(ss->ldQualifier);
        auto vecAddr = vec.front()->vec;
        vec.pop();
        assert(vecAddr.size() == 2);
        for (int i = 0; i < 2; i++) {
            to = vecAddr[i];
            mov((void *)((uint64_t)from + i * step), to, ss->ldQualifier);
        }
    } else if (QvecHasQ(ss->ldQualifier, Qualifier::Q_V4)) {
        uint64_t step = getBytes(ss->ldQualifier);
        auto vecAddr = vec.front()->vec;
        vec.pop();
        assert(vecAddr.size() == 4);
        for (int i = 0; i < 4; i++) {
            to = vecAddr[i];
            mov((void *)((uint64_t)from + i * step), to, ss->ldQualifier);
        }
    } else {
        mov(from, to, ss->ldQualifier);
    }
}

void ThreadContext::handle_mov(StatementContext::MOV *ss) {
    // op0
    void *to = getOperandAddr(ss->movOp[0], ss->movQualifier);

    // op1
    void *from = getOperandAddr(ss->movOp[1], ss->movQualifier);

    // exe mov
    mov(from, to, ss->movQualifier);
}

void ThreadContext::handle_add(StatementContext::ADD *ss) {
    // op0
    void *to = getOperandAddr(ss->addOp[0], ss->addQualifier);

    // op1
    void *op1 = getOperandAddr(ss->addOp[1], ss->addQualifier);

    // op2
    void *op2 = getOperandAddr(ss->addOp[2], ss->addQualifier);

    // exe add
    add(to, op1, op2, ss->addQualifier);
}

void ThreadContext::handle_sub(StatementContext::SUB *ss) {
    // op0
    void *to = getOperandAddr(ss->subOp[0], ss->subQualifier);

    // op1
    void *op1 = getOperandAddr(ss->subOp[1], ss->subQualifier);

    // op2
    void *op2 = getOperandAddr(ss->subOp[2], ss->subQualifier);

    // exe sub
    sub(to, op1, op2, ss->subQualifier);
}

void ThreadContext::handle_mul(StatementContext::MUL *ss) {
    // op0
    void *to = getOperandAddr(ss->mulOp[0], ss->mulQualifier);

    // op1
    void *op1 = getOperandAddr(ss->mulOp[1], ss->mulQualifier);

    // op2
    void *op2 = getOperandAddr(ss->mulOp[2], ss->mulQualifier);

    // exe mul
    mul(to, op1, op2, ss->mulQualifier);
}

void ThreadContext::handle_div(StatementContext::DIV *ss) {
    // op0
    void *to = getOperandAddr(ss->divOp[0], ss->divQualifier);

    // op1
    void *op1 = getOperandAddr(ss->divOp[1], ss->divQualifier);

    // op2
    void *op2 = getOperandAddr(ss->divOp[2], ss->divQualifier);

    // exe div
    div(to, op1, op2, ss->divQualifier);
}

void ThreadContext::handle_and(StatementContext::AND *ss) {
    // op0
    void *to = getOperandAddr(ss->andOp[0], ss->andQualifier);

    // op1
    void *op1 = getOperandAddr(ss->andOp[1], ss->andQualifier);

    // op2
    void *op2 = getOperandAddr(ss->andOp[2], ss->andQualifier);

    // exe and
    And(to, op1, op2, ss->andQualifier);
}

void ThreadContext::handle_or(StatementContext::OR *ss) {
    // op0
    void *to = getOperandAddr(ss->orOp[0], ss->orQualifier);

    // op1
    void *op1 = getOperandAddr(ss->orOp[1], ss->orQualifier);

    // op2
    void *op2 = getOperandAddr(ss->orOp[2], ss->orQualifier);

    // exe or
    Or(to, op1, op2, ss->orQualifier);
}

void ThreadContext::handle_xor(StatementContext::XOR *ss) {
    // op0
    void *to = getOperandAddr(ss->xorOp[0], ss->xorQualifier);

    // op1
    void *op1 = getOperandAddr(ss->xorOp[1], ss->xorQualifier);

    // op2
    void *op2 = getOperandAddr(ss->xorOp[2], ss->xorQualifier);

    // exe xor
    Xor(to, op1, op2, ss->xorQualifier);
}

void ThreadContext::handle_shl(StatementContext::SHL *ss) {
    // op0
    void *to = getOperandAddr(ss->shlOp[0], ss->shlQualifier);

    // op1
    void *op1 = getOperandAddr(ss->shlOp[1], ss->shlQualifier);

    // op2
    std::vector<Qualifier> tq;
    tq.push_back(Qualifier::Q_U32);
    void *op2 = getOperandAddr(ss->shlOp[2], tq);

    // exe shl
    shl(to, op1, op2, ss->shlQualifier);
}

void ThreadContext::handle_shr(StatementContext::SHR *ss) {
    // op0
    void *to = getOperandAddr(ss->shrOp[0], ss->shrQualifier);

    // op1
    void *op1 = getOperandAddr(ss->shrOp[1], ss->shrQualifier);

    // op2
    std::vector<Qualifier> tq;
    tq.push_back(Qualifier::Q_U32);
    void *op2 = getOperandAddr(ss->shrOp[2], tq);

    // exe shr
    shr(to, op1, op2, ss->shrQualifier);
}

void ThreadContext::handle_st(StatementContext::ST *ss) {
    // op0
    void *to = getOperandAddr(ss->stOp[0], ss->stQualifier);

    // op1
    void *from = getOperandAddr(ss->stOp[1], ss->stQualifier);

    // exe st
    if (QvecHasQ(ss->stQualifier, Qualifier::Q_V4)) {
        uint64_t step = getBytes(ss->stQualifier);
        auto vecAddr = vec.front()->vec;
        vec.pop();
        assert(vecAddr.size() == 4);
        for (int i = 0; i < 4; i++) {
            from = vecAddr[i];
            mov(from, (void *)((uint64_t)to + i * step), ss->stQualifier);
        }
    } else if (QvecHasQ(ss->stQualifier, Qualifier::Q_V2)) {
        uint64_t step = getBytes(ss->stQualifier);
        auto vecAddr = vec.front()->vec;
        vec.pop();
        assert(vecAddr.size() == 2);
        for (int i = 0; i < 2; i++) {
            from = vecAddr[i];
            mov(from, (void *)((uint64_t)to + i * step), ss->stQualifier);
        }
    } else {
        mov(from, to, ss->stQualifier);
    }
}

void ThreadContext::handle_selp(StatementContext::SELP *ss) {
    // op0
    void *to = getOperandAddr(ss->selpOp[0], ss->selpQualifier);

    // op1
    void *op0 = getOperandAddr(ss->selpOp[1], ss->selpQualifier);

    // op2
    void *op1 = getOperandAddr(ss->selpOp[2], ss->selpQualifier);

    // op3
    void *pred = getOperandAddr(ss->selpOp[3], ss->selpQualifier);

    // exe selp
    this->selp(to, op0, op1, pred, ss->selpQualifier);
}

void ThreadContext::handle_sin(StatementContext::SIN *ss) {
    // op0
    void *to = getOperandAddr(ss->sinOp[0], ss->sinQualifier);

    // op1
    void *op = getOperandAddr(ss->sinOp[1], ss->sinQualifier);

    // exe sin
    this->m_sin(to, op, ss->sinQualifier);
}

void ThreadContext::handle_cos(StatementContext::COS *ss) {
    // op0
    void *to = getOperandAddr(ss->cosOp[0], ss->cosQualifier);

    // op1
    void *op = getOperandAddr(ss->cosOp[1], ss->cosQualifier);

    // exe cos
    this->m_cos(to, op, ss->cosQualifier);
}

void ThreadContext::handle_lg2(StatementContext::LG2 *ss) {
    // op0
    void *to = getOperandAddr(ss->lg2Op[0], ss->lg2Qualifier);

    // op1
    void *op = getOperandAddr(ss->lg2Op[1], ss->lg2Qualifier);

    // exe lg2
    this->m_lg2(to, op, ss->lg2Qualifier);
}

void ThreadContext::handle_ex2(StatementContext::EX2 *ss) {
    // op0
    void *to = getOperandAddr(ss->ex2Op[0], ss->ex2Qualifier);

    // op1
    void *op = getOperandAddr(ss->ex2Op[1], ss->ex2Qualifier);

    // exe ex2
    this->m_ex2(to, op, ss->ex2Qualifier);
}

void ThreadContext::handle_rem(StatementContext::REM *ss) {
    // op0
    void *to = getOperandAddr(ss->remOp[0], ss->remQualifier);

    // op1
    void *op1 = getOperandAddr(ss->remOp[1], ss->remQualifier);

    // op2
    void *op2 = getOperandAddr(ss->remOp[2], ss->remQualifier);

    // exe rem
    this->m_rem(to, op1, op2, ss->remQualifier);
}

void ThreadContext::handle_cvta(StatementContext::CVTA *ss) {
    // op0
    void *to = getOperandAddr(ss->cvtaOp[0], ss->cvtaQualifier);

    // op1
    void *op = getOperandAddr(ss->cvtaOp[1], ss->cvtaQualifier);

    // exe cvta
    mov(op, to, ss->cvtaQualifier);
}

void ThreadContext::handle_not(StatementContext::NOT *ss) {
    /* Empty implementation to avoid linker error */
}

void ThreadContext::handle_rsqrt(StatementContext::RSQRT *ss) {
    // op0
    void *to = getOperandAddr(ss->rsqrtOp[0], ss->rsqrtQualifier);

    // op1
    void *op = getOperandAddr(ss->rsqrtOp[1], ss->rsqrtQualifier);

    // exe rsqrt
    this->m_rsqrt(to, op, ss->rsqrtQualifier);
}

void ThreadContext::handle_wmma(
    StatementContext::WMMA *ss) { /* Not implemented yet */ }

// Comparison operator helper
Qualifier ThreadContext::getCMPOP(std::vector<Qualifier> &q) {
    for (auto e : q) {
        switch (e) {
        case Qualifier::Q_EQ:
        case Qualifier::Q_NE:
        case Qualifier::Q_LT:
        case Qualifier::Q_LE:
        case Qualifier::Q_GT:
        case Qualifier::Q_GE:
        case Qualifier::Q_LO:
        case Qualifier::Q_HI:
        case Qualifier::Q_LTU:
        case Qualifier::Q_LEU:
        case Qualifier::Q_GEU:
        case Qualifier::Q_NEU:
        case Qualifier::Q_GTU:
            return e;
        default:
            continue;
        }
    }
    return Qualifier::Q_EQ; // Default comparison operator
}

// Float vs Integer type detection
ThreadContext::DTYPE ThreadContext::getDType(std::vector<Qualifier> &q) {
    for (auto e : q) {
        switch (e) {
        case Qualifier::Q_F64:
        case Qualifier::Q_F32:
        case Qualifier::Q_F16:
        case Qualifier::Q_F8:
            return DTYPE::DFLOAT;
        case Qualifier::Q_S64:
        case Qualifier::Q_B64:
        case Qualifier::Q_U64:
        case Qualifier::Q_S32:
        case Qualifier::Q_B32:
        case Qualifier::Q_U32:
        case Qualifier::Q_S16:
        case Qualifier::Q_B16:
        case Qualifier::Q_U16:
        case Qualifier::Q_S8:
        case Qualifier::Q_B8:
        case Qualifier::Q_U8:
        case Qualifier::Q_PRED:
            return DTYPE::DINT;
        default:
            continue;
        }
    }
    return DTYPE::DINT; // Default to integer
}

ThreadContext::DTYPE ThreadContext::getDType(Qualifier q) {
    switch (q) {
    case Qualifier::Q_F64:
    case Qualifier::Q_F32:
    case Qualifier::Q_F16:
    case Qualifier::Q_F8:
        return DTYPE::DFLOAT;
    case Qualifier::Q_S64:
    case Qualifier::Q_B64:
    case Qualifier::Q_U64:
    case Qualifier::Q_S32:
    case Qualifier::Q_B32:
    case Qualifier::Q_U32:
    case Qualifier::Q_S16:
    case Qualifier::Q_B16:
    case Qualifier::Q_U16:
    case Qualifier::Q_S8:
    case Qualifier::Q_B8:
    case Qualifier::Q_U8:
        return DTYPE::DINT;
    default:
        return DTYPE::DNONE;
    }
}

bool ThreadContext::Signed(Qualifier q) {
    switch (q) {
    case Qualifier::Q_S64:
    case Qualifier::Q_S32:
    case Qualifier::Q_S16:
    case Qualifier::Q_S8:
        return true;
    default:
        return false;
    }
}

// Template implementations for various operations

template <typename T> void _setp_eq(void *to, void *op1, void *op2) {
    *(uint8_t *)to = *(T *)op1 == *(T *)op2;
}

template <typename T> void _setp_ne(void *to, void *op1, void *op2, bool mask) {
    if (*(T *)op1 != *(T *)op1 || *(T *)op2 != *(T *)op2)
        *(uint8_t *)to = mask;
    else
        *(uint8_t *)to = *(T *)op1 != *(T *)op2;
}

template <typename T> void _setp_lt(void *to, void *op1, void *op2, bool mask) {
    if (*(T *)op1 != *(T *)op1 || *(T *)op2 != *(T *)op2)
        *(uint8_t *)to = mask;
    else
        *(uint8_t *)to = *(T *)op1 < *(T *)op2;
}

template <typename T> void _setp_le(void *to, void *op1, void *op2, bool mask) {
    if (*(T *)op1 != *(T *)op1 || *(T *)op2 != *(T *)op2)
        *(uint8_t *)to = mask;
    else
        *(uint8_t *)to = *(T *)op1 <= *(T *)op2;
}

template <typename T> void _setp_ge(void *to, void *op1, void *op2, bool mask) {
    if (*(T *)op1 != *(T *)op1 || *(T *)op2 != *(T *)op2)
        *(uint8_t *)to = mask;
    else
        *(uint8_t *)to = *(T *)op1 >= *(T *)op2;
}

template <typename T> void _setp_gt(void *to, void *op1, void *op2, bool mask) {
    if (*(T *)op1 != *(T *)op1 || *(T *)op2 != *(T *)op2)
        *(uint8_t *)to = mask;
    else
        *(uint8_t *)to = *(T *)op1 > *(T *)op2;
}

template <typename T>
void _setp(void *to, void *op1, void *op2, Qualifier cmpOp) {
    bool res = false;

    switch (cmpOp) {
    case Qualifier::Q_EQ:
        res = *(T *)op1 == *(T *)op2;
        break;
    case Qualifier::Q_NE:
        res = *(T *)op1 != *(T *)op2;
        break;
    case Qualifier::Q_LT:
        res = *(T *)op1 < *(T *)op2;
        break;
    case Qualifier::Q_LE:
        res = *(T *)op1 <= *(T *)op2;
        break;
    case Qualifier::Q_GT:
        res = *(T *)op1 > *(T *)op2;
        break;
    case Qualifier::Q_GE:
        res = *(T *)op1 >= *(T *)op2;
        break;
    default:
        assert(0);
    }

    *(bool *)to = res;
}

void ThreadContext::setp(void *to, void *op1, void *op2, Qualifier cmpOp,
                         std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        assert(dtype == DTYPE::DINT);
        _setp<uint8_t>(to, op1, op2, cmpOp);
        return;
    }
    case 2: {
        assert(dtype == DTYPE::DINT);
        _setp<uint16_t>(to, op1, op2, cmpOp);
        return;
    }
    case 4: {
        switch (dtype) {
        case DTYPE::DINT:
            _setp<uint32_t>(to, op1, op2, cmpOp);
            return;
        case DTYPE::DFLOAT:
            _setp<float>(to, op1, op2, cmpOp);
            return;
        default:
            assert(0);
        }
        return;
    }
    case 8: {
        switch (dtype) {
        case DTYPE::DINT:
            _setp<uint64_t>(to, op1, op2, cmpOp);
            return;
        case DTYPE::DFLOAT:
            _setp<double>(to, op1, op2, cmpOp);
            return;
        default:
            assert(0);
        }
    }
    default:
        assert(0);
    }
}

void ThreadContext::cvt(void *to, void *from, std::vector<Qualifier> &toQ,
                        std::vector<Qualifier> &fromQ) {
    int toLen = getBytes(toQ);
    int fromLen = getBytes(fromQ);
    DTYPE toDtype = getDType(toQ);
    DTYPE fromDtype = getDType(fromQ);

    // Same size and type conversion (just copy)
    if (toLen == fromLen && toDtype == fromDtype) {
        mov(from, to, toQ);
        return;
    }

    // Integer to integer conversion
    if (toDtype == DTYPE::DINT && fromDtype == DTYPE::DINT) {
        switch (toLen) {
        case 1:
            switch (fromLen) {
            case 1:
                *(uint8_t *)to = *(uint8_t *)from;
                break;
            case 2:
                *(uint8_t *)to = *(uint16_t *)from;
                break;
            case 4:
                *(uint8_t *)to = *(uint32_t *)from;
                break;
            case 8:
                *(uint8_t *)to = *(uint64_t *)from;
                break;
            default:
                assert(0);
            }
            break;
        case 2:
            switch (fromLen) {
            case 1:
                *(uint16_t *)to = *(uint8_t *)from;
                break;
            case 2:
                *(uint16_t *)to = *(uint16_t *)from;
                break;
            case 4:
                *(uint16_t *)to = *(uint32_t *)from;
                break;
            case 8:
                *(uint16_t *)to = *(uint64_t *)from;
                break;
            default:
                assert(0);
            }
            break;
        case 4:
            switch (fromLen) {
            case 1:
                *(uint32_t *)to = *(uint8_t *)from;
                break;
            case 2:
                *(uint32_t *)to = *(uint16_t *)from;
                break;
            case 4:
                *(uint32_t *)to = *(uint32_t *)from;
                break;
            case 8:
                *(uint32_t *)to = *(uint64_t *)from;
                break;
            default:
                assert(0);
            }
            break;
        case 8:
            switch (fromLen) {
            case 1:
                *(uint64_t *)to = *(uint8_t *)from;
                break;
            case 2:
                *(uint64_t *)to = *(uint16_t *)from;
                break;
            case 4:
                *(uint64_t *)to = *(uint32_t *)from;
                break;
            case 8:
                *(uint64_t *)to = *(uint64_t *)from;
                break;
            default:
                assert(0);
            }
            break;
        default:
            assert(0);
        }
        return;
    }

    // Float to float conversion
    if (toDtype == DTYPE::DFLOAT && fromDtype == DTYPE::DFLOAT) {
        switch (toLen) {
        case 4:
            switch (fromLen) {
            case 4:
                *(float *)to = *(float *)from;
                break;
            case 8:
                *(float *)to = *(double *)from;
                break;
            default:
                assert(0);
            }
            break;
        case 8:
            switch (fromLen) {
            case 4:
                *(double *)to = *(float *)from;
                break;
            case 8:
                *(double *)to = *(double *)from;
                break;
            default:
                assert(0);
            }
            break;
        default:
            assert(0);
        }
        return;
    }

    // Int to float conversion
    if (toDtype == DTYPE::DFLOAT && fromDtype == DTYPE::DINT) {
        switch (toLen) {
        case 4:
            switch (fromLen) {
            case 1:
                *(float *)to = *(uint8_t *)from;
                break;
            case 2:
                *(float *)to = *(uint16_t *)from;
                break;
            case 4:
                *(float *)to = *(uint32_t *)from;
                break;
            case 8:
                *(float *)to = *(uint64_t *)from;
                break;
            default:
                assert(0);
            }
            break;
        case 8:
            switch (fromLen) {
            case 1:
                *(double *)to = *(uint8_t *)from;
                break;
            case 2:
                *(double *)to = *(uint16_t *)from;
                break;
            case 4:
                *(double *)to = *(uint32_t *)from;
                break;
            case 8:
                *(double *)to = *(uint64_t *)from;
                break;
            default:
                assert(0);
            }
            break;
        default:
            assert(0);
        }
        return;
    }

    // Float to int conversion
    if (toDtype == DTYPE::DINT && fromDtype == DTYPE::DFLOAT) {
        switch (toLen) {
        case 1:
            switch (fromLen) {
            case 4:
                *(uint8_t *)to = *(float *)from;
                break;
            case 8:
                *(uint8_t *)to = *(double *)from;
                break;
            default:
                assert(0);
            }
            break;
        case 2:
            switch (fromLen) {
            case 4:
                *(uint16_t *)to = *(float *)from;
                break;
            case 8:
                *(uint16_t *)to = *(double *)from;
                break;
            default:
                assert(0);
            }
            break;
        case 4:
            switch (fromLen) {
            case 4:
                *(uint32_t *)to = *(float *)from;
                break;
            case 8:
                *(uint32_t *)to = *(double *)from;
                break;
            default:
                assert(0);
            }
            break;
        case 8:
            switch (fromLen) {
            case 4:
                *(uint64_t *)to = *(float *)from;
                break;
            case 8:
                *(uint64_t *)to = *(double *)from;
                break;
            default:
                assert(0);
            }
            break;
        default:
            assert(0);
        }
        return;
    }

    assert(0);
}

template <typename T> void _max(void *to, void *op1, void *op2) {
    *(T *)to = std::max(*(T *)op1, *(T *)op2);
}

void ThreadContext::max(void *to, void *op1, void *op2,
                        std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _max<uint8_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _max<uint16_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 4: {
        switch (dtype) {
        case DTYPE::DINT:
            _max<uint32_t>(to, op1, op2);
            return;
        case DTYPE::DFLOAT:
            _max<float>(to, op1, op2);
            return;
        default:
            assert(0);
        }
        return;
    }
    case 8: {
        switch (dtype) {
        case DTYPE::DINT:
            _max<uint64_t>(to, op1, op2);
            return;
        case DTYPE::DFLOAT:
            _max<double>(to, op1, op2);
            return;
        default:
            assert(0);
        }
    }
    default:
        assert(0);
    }
}

template <typename T> void _min(void *to, void *op1, void *op2) {
    *(T *)to = std::min(*(T *)op1, *(T *)op2);
}

void ThreadContext::min(void *to, void *op1, void *op2,
                        std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _min<uint8_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _min<uint16_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 4: {
        switch (dtype) {
        case DTYPE::DINT:
            _min<uint32_t>(to, op1, op2);
            return;
        case DTYPE::DFLOAT:
            _min<float>(to, op1, op2);
            return;
        default:
            assert(0);
        }
        return;
    }
    case 8: {
        switch (dtype) {
        case DTYPE::DINT:
            _min<uint64_t>(to, op1, op2);
            return;
        case DTYPE::DFLOAT:
            _min<double>(to, op1, op2);
            return;
        default:
            assert(0);
        }
    }
    default:
        assert(0);
    }
}

template <typename T> void _mad(void *to, void *op1, void *op2, void *op3) {
    *(T *)to = (*(T *)op1) * (*(T *)op2) + (*(T *)op3);
}

void ThreadContext::mad(void *to, void *op1, void *op2, void *op3,
                        std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _mad<uint8_t>(to, op1, op2, op3);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _mad<uint16_t>(to, op1, op2, op3);
        else
            assert(0);
        return;
    }
    case 4: {
        switch (dtype) {
        case DTYPE::DINT:
            _mad<uint32_t>(to, op1, op2, op3);
            return;
        case DTYPE::DFLOAT:
            _mad<float>(to, op1, op2, op3);
            return;
        default:
            assert(0);
        }
        return;
    }
    case 8: {
        switch (dtype) {
        case DTYPE::DINT:
            _mad<uint64_t>(to, op1, op2, op3);
            return;
        case DTYPE::DFLOAT:
            _mad<double>(to, op1, op2, op3);
            return;
        default:
            assert(0);
        }
    }
    default:
        assert(0);
    }
}

template <typename T> void _fma(void *to, void *op1, void *op2, void *op3) {
    *(T *)to = std::fma(*(T *)op1, *(T *)op2, *(T *)op3);
}

void ThreadContext::fma(void *to, void *op1, void *op2, void *op3,
                        std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 4: {
        if (dtype == DTYPE::DFLOAT)
            _fma<float>(to, op1, op2, op3);
        else
            assert(0);
        return;
    }
    case 8: {
        if (dtype == DTYPE::DFLOAT)
            _fma<double>(to, op1, op2, op3);
        else
            assert(0);
        return;
    }
    default:
        assert(0);
    }
}

template <typename T> void _neg(void *to, void *op) { *(T *)to = -(*(T *)op); }

void ThreadContext::neg(void *to, void *op, std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _neg<uint8_t>(to, op);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _neg<uint16_t>(to, op);
        else
            assert(0);
        return;
    }
    case 4: {
        switch (dtype) {
        case DTYPE::DINT:
            _neg<uint32_t>(to, op);
            return;
        case DTYPE::DFLOAT:
            _neg<float>(to, op);
            return;
        default:
            assert(0);
        }
        return;
    }
    case 8: {
        switch (dtype) {
        case DTYPE::DINT:
            _neg<uint64_t>(to, op);
            return;
        case DTYPE::DFLOAT:
            _neg<double>(to, op);
            return;
        default:
            assert(0);
        }
    }
    default:
        assert(0);
    }
}

template <typename T> void _sqrt(void *to, void *op) {
    *(T *)to = std::sqrt(*(T *)op);
}

void ThreadContext::sqrt(void *to, void *op, std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 4: {
        if (dtype == DTYPE::DFLOAT)
            _sqrt<float>(to, op);
        else
            assert(0);
        return;
    }
    case 8: {
        if (dtype == DTYPE::DFLOAT)
            _sqrt<double>(to, op);
        else
            assert(0);
        return;
    }
    default:
        assert(0);
    }
}

void ThreadContext::atom(void *to, void *op, std::vector<Qualifier> &q) {
    // TODO: Implement atomic operations
    /* Not implemented yet */
}

template <typename T> void _abs(void *to, void *op) {
    if constexpr (std::is_unsigned_v<T>) {
        *(T *)to = *(T *)op;
    } else {
        *(T *)to = std::abs(*(T *)op);
    }
}

void ThreadContext::abs(void *to, void *op, std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _abs<uint8_t>(to, op);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _abs<uint16_t>(to, op);
        else
            assert(0);
        return;
    }
    case 4: {
        switch (dtype) {
        case DTYPE::DINT:
            _abs<uint32_t>(to, op);
            return;
        case DTYPE::DFLOAT:
            _abs<float>(to, op);
            return;
        default:
            assert(0);
        }
        return;
    }
    case 8: {
        switch (dtype) {
        case DTYPE::DINT:
            _abs<uint64_t>(to, op);
            return;
        case DTYPE::DFLOAT:
            _abs<double>(to, op);
            return;
        default:
            assert(0);
        }
    }
    default:
        assert(0);
    }
}

template <typename T> void _rcp(void *to, void *from) {
    *(T *)to = 1.0 / *(T *)from;
}

void ThreadContext::rcp(void *to, void *from, std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);

    switch (len) {
    case 4:
        if (dtype == DTYPE::DFLOAT) {
            *(float *)to = 1.0f / *(float *)from;
        } else {
            assert(0);
        }
        break;
    case 8:
        if (dtype == DTYPE::DFLOAT) {
            *(double *)to = 1.0 / *(double *)from;
        } else {
            assert(0);
        }
        break;
    default:
        assert(0);
    }
}

template <typename T> void _sin(void *to, void *op) {
    *(T *)to = std::sin(*(T *)op);
}

void ThreadContext::m_sin(void *to, void *op, std::vector<Qualifier> &q) {
    DTYPE dtype = getDType(q);
    int len = getBytes(q);

    switch (len) {
    case 4:
        _sin<float>(to, op);
        break;
    case 8:
        _sin<double>(to, op);
        break;
    default:
        assert(0);
    }
}

template <typename T> void _cos(void *to, void *op) {
    *(T *)to = std::cos(*(T *)op);
}

void ThreadContext::m_cos(void *to, void *op, std::vector<Qualifier> &q) {
    DTYPE dtype = getDType(q);
    int len = getBytes(q);

    switch (len) {
    case 4:
        _cos<float>(to, op);
        break;
    case 8:
        _cos<double>(to, op);
        break;
    default:
        assert(0);
    }
}

template <typename T> void _lg2(void *to, void *op) {
    *(T *)to = std::log2(*(T *)op);
}

void ThreadContext::m_lg2(void *to, void *op, std::vector<Qualifier> &q) {
    DTYPE dtype = getDType(q);
    int len = getBytes(q);

    switch (len) {
    case 4:
        _lg2<float>(to, op);
        break;
    case 8:
        _lg2<double>(to, op);
        break;
    default:
        assert(0);
    }
}

template <typename T> void _ex2(void *to, void *op) {
    *(T *)to = std::exp2(*(T *)op);
}

void ThreadContext::m_ex2(void *to, void *op, std::vector<Qualifier> &q) {
    DTYPE dtype = getDType(q);
    int len = getBytes(q);

    switch (len) {
    case 4:
        _ex2<float>(to, op);
        break;
    case 8:
        _ex2<double>(to, op);
        break;
    default:
        assert(0);
    }
}

template <typename T> void _rem(void *to, void *op1, void *op2) {
    *(T *)to = *(T *)op1 % *(T *)op2;
}

void ThreadContext::m_rem(void *to, void *op1, void *op2,
                          std::vector<Qualifier> &q) {
    int len = getBytes(q);
    Qualifier datatype = getDataType(q);
    switch (len) {
    case 1: {
        if (Signed(datatype))
            _rem<int8_t>(to, op1, op2);
        else
            _rem<uint8_t>(to, op1, op2);
        return;
    }
    case 2: {
        if (Signed(datatype))
            _rem<int16_t>(to, op1, op2);
        else
            _rem<uint16_t>(to, op1, op2);
        return;
    }
    case 4: {
        if (Signed(datatype))
            _rem<int32_t>(to, op1, op2);
        else
            _rem<uint32_t>(to, op1, op2);
        return;
    }
    case 8: {
        if (Signed(datatype))
            _rem<int64_t>(to, op1, op2);
        else
            _rem<uint64_t>(to, op1, op2);
        return;
    }
    default:
        assert(0);
    }
}

template <typename T> void _rsqrt(void *to, void *op) {
    *(T *)to = 1.0 / std::sqrt(*(T *)op);
}

void ThreadContext::m_rsqrt(void *to, void *op, std::vector<Qualifier> &q) {
    DTYPE dtype = getDType(q);
    int len = getBytes(q);

    switch (len) {
    case 4:
        _rsqrt<float>(to, op);
        break;
    case 8:
        _rsqrt<double>(to, op);
        break;
    default:
        assert(0);
    }
}

template <typename T> void _add(void *to, void *op1, void *op2) {
    *(T *)to = (*(T *)op1) + (*(T *)op2);
}

void ThreadContext::add(void *to, void *op1, void *op2,
                        std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _add<uint8_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _add<uint16_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 4: {
        switch (dtype) {
        case DTYPE::DINT:
            _add<uint32_t>(to, op1, op2);
            return;
        case DTYPE::DFLOAT:
            _add<float>(to, op1, op2);
            return;
        default:
            assert(0);
        }
        return;
    }
    case 8: {
        switch (dtype) {
        case DTYPE::DINT:
            _add<uint64_t>(to, op1, op2);
            return;
        case DTYPE::DFLOAT:
            _add<double>(to, op1, op2);
            return;
        default:
            assert(0);
        }
    }
    default:
        assert(0);
    }
}

template <typename T> void _sub(void *to, void *op1, void *op2) {
    *(T *)to = (*(T *)op1) - (*(T *)op2);
}

void ThreadContext::sub(void *to, void *op1, void *op2,
                        std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _sub<uint8_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _sub<uint16_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 4: {
        switch (dtype) {
        case DTYPE::DINT:
            _sub<uint32_t>(to, op1, op2);
            return;
        case DTYPE::DFLOAT:
            _sub<float>(to, op1, op2);
            return;
        default:
            assert(0);
        }
        return;
    }
    case 8: {
        switch (dtype) {
        case DTYPE::DINT:
            _sub<uint64_t>(to, op1, op2);
            return;
        case DTYPE::DFLOAT:
            _sub<double>(to, op1, op2);
            return;
        default:
            assert(0);
        }
    }
    default:
        assert(0);
    }
}

template <typename T> void _mul(void *to, void *op1, void *op2) {
    *(T *)to = (*(T *)op1) * (*(T *)op2);
}

void ThreadContext::mul(void *to, void *op1, void *op2,
                        std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _mul<uint8_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _mul<uint16_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 4: {
        switch (dtype) {
        case DTYPE::DINT:
            _mul<uint32_t>(to, op1, op2);
            return;
        case DTYPE::DFLOAT:
            _mul<float>(to, op1, op2);
            return;
        default:
            assert(0);
        }
        return;
    }
    case 8: {
        switch (dtype) {
        case DTYPE::DINT:
            _mul<uint64_t>(to, op1, op2);
            return;
        case DTYPE::DFLOAT:
            _mul<double>(to, op1, op2);
            return;
        default:
            assert(0);
        }
    }
    default:
        assert(0);
    }
}

template <typename T> void _div(void *to, void *op1, void *op2) {
    *(T *)to = (*(T *)op1) / (*(T *)op2);
}

void ThreadContext::div(void *to, void *op1, void *op2,
                        std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _div<uint8_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _div<uint16_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 4: {
        switch (dtype) {
        case DTYPE::DINT:
            _div<uint32_t>(to, op1, op2);
            return;
        case DTYPE::DFLOAT:
            _div<float>(to, op1, op2);
            return;
        default:
            assert(0);
        }
        return;
    }
    case 8: {
        switch (dtype) {
        case DTYPE::DINT:
            _div<uint64_t>(to, op1, op2);
            return;
        case DTYPE::DFLOAT:
            _div<double>(to, op1, op2);
            return;
        default:
            assert(0);
        }
    }
    default:
        assert(0);
    }
}

template <typename T> void _and(void *to, void *op1, void *op2) {
    *(T *)to = (*(T *)op1) & (*(T *)op2);
}

void ThreadContext::And(void *to, void *op1, void *op2,
                        std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _and<uint8_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _and<uint16_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 4: {
        if (dtype == DTYPE::DINT)
            _and<uint32_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 8: {
        if (dtype == DTYPE::DINT)
            _and<uint64_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    default:
        assert(0);
    }
}

template <typename T> void _or(void *to, void *op1, void *op2) {
    *(T *)to = (*(T *)op1) | (*(T *)op2);
}

void ThreadContext::Or(void *to, void *op1, void *op2,
                       std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _or<uint8_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _or<uint16_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 4: {
        if (dtype == DTYPE::DINT)
            _or<uint32_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 8: {
        if (dtype == DTYPE::DINT)
            _or<uint64_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    default:
        assert(0);
    }
}

template <typename T> void _xor(void *to, void *op1, void *op2) {
    *(T *)to = (*(T *)op1) ^ (*(T *)op2);
}

void ThreadContext::Xor(void *to, void *op1, void *op2,
                        std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _xor<uint8_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _xor<uint16_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 4: {
        if (dtype == DTYPE::DINT)
            _xor<uint32_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 8: {
        if (dtype == DTYPE::DINT)
            _xor<uint64_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    default:
        assert(0);
    }
}

template <typename T> void _shl(void *to, void *op1, void *op2) {
    *(T *)to = (*(T *)op1) << (*(T *)op2);
}

void ThreadContext::shl(void *to, void *op1, void *op2,
                        std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _shl<uint8_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _shl<uint16_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 4: {
        if (dtype == DTYPE::DINT)
            _shl<uint32_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 8: {
        if (dtype == DTYPE::DINT)
            _shl<uint64_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    default:
        assert(0);
    }
}

template <typename T> void _shr(void *to, void *op1, void *op2) {
    *(T *)to = (*(T *)op1) >> (*(T *)op2);
}

void ThreadContext::shr(void *to, void *op1, void *op2,
                        std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1: {
        if (dtype == DTYPE::DINT)
            _shr<uint8_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 2: {
        if (dtype == DTYPE::DINT)
            _shr<uint16_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 4: {
        if (dtype == DTYPE::DINT)
            _shr<uint32_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    case 8: {
        if (dtype == DTYPE::DINT)
            _shr<uint64_t>(to, op1, op2);
        else
            assert(0);
        return;
    }
    default:
        assert(0);
    }
}

template <typename T> void _selp(void *to, void *op0, void *op1, void *pred) {
    *(T *)to = *(bool *)pred ? *(T *)op0 : *(T *)op1;
}

void ThreadContext::selp(void *to, void *op0, void *op1, void *pred,
                         std::vector<Qualifier> &q) {
    int len = getBytes(q);
    DTYPE dtype = getDType(q);
    switch (len) {
    case 1:
        assert(dtype == DTYPE::DINT);
        _selp<uint8_t>(to, op0, op1, pred);
        return;
    case 2:
        assert(dtype == DTYPE::DINT);
        _selp<uint16_t>(to, op0, op1, pred);
        return;
    case 4:
        switch (dtype) {
        case DTYPE::DINT:
            _selp<uint32_t>(to, op0, op1, pred);
            return;
        case DTYPE::DFLOAT:
            _selp<float>(to, op0, op1, pred);
            return;
        default:
            assert(0);
        }
        return;
    case 8:
        switch (dtype) {
        case DTYPE::DINT:
            _selp<uint64_t>(to, op0, op1, pred);
            return;
        case DTYPE::DFLOAT:
            _selp<double>(to, op0, op1, pred);
            return;
        default:
            assert(0);
        }
    default:
        assert(0);
    }
}
