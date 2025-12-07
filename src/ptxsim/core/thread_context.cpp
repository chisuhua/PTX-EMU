#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/interpreter.h"
#include "ptxsim/instruction_factory.h"
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

EXE_STATE ThreadContext::exe_once() {
    if (state != RUN) return state;
    
    _execute_once();
    
    if (pc >= statements->size()) 
        state = EXIT;
        
    clear_temporaries();
    return state;
}

void ThreadContext::_execute_once() {
    // 获取当前语句
    StatementContext& stmt = (*statements)[pc];
    
    // 使用工厂创建对应的处理器并执行
    InstructionHandler* handler = InstructionFactory::create_handler(stmt.statementType);
    if (handler) {
        handler->execute(this, stmt);
        pc++;
    } else {
        std::cerr << "No handler found for statement type: " << static_cast<int>(stmt.statementType) << std::endl;
        state = EXIT;
    }
}

void ThreadContext::clear_temporaries() {
    while (!imm.empty()) {
        delete imm.front();
        imm.pop();
    }
    while (!vec.empty()) {
        delete vec.front();
        vec.pop();
    }
}

void* ThreadContext::get_operand_addr(OperandContext &op, std::vector<Qualifier> &qualifiers) {
    switch (op.operandType) {
        case O_REG:
            return get_register_addr((OperandContext::REG*)op.operand);
            
        case O_FA:
            return get_memory_addr((OperandContext::FA*)op.operand, qualifiers);
            
        case O_IMM:
            {
                auto immOp = (OperandContext::IMM*)op.operand;
                // 创建一个新的IMM对象用于临时存储
                PtxInterpreter::IMM* newImm = new PtxInterpreter::IMM();
                // TODO: 需要根据类型正确设置IMM的值
                imm.push(newImm);
                return &(newImm->data);
            }
            
        case O_VEC:
            {
                auto vecOp = (OperandContext::VEC*)op.operand;
                // 创建一个新的VEC对象用于临时存储
                PtxInterpreter::VEC* newVec = new PtxInterpreter::VEC();
                // TODO: 需要正确设置VEC的值
                vec.push(newVec);
                return &(newVec->vec);
            }
            
        default:
            assert(false && "Unknown operand type");
            return nullptr;
    }
}

void* ThreadContext::get_register_addr(OperandContext::REG *reg) {
    std::string regName = reg->regName;
    
    // 查找寄存器
    auto iter = name2Reg.find(regName);
    if (iter != name2Reg.end()) {
        return iter->second->addr;
    }
    
    // 如果没找到，可能是共享内存或局部内存
    auto shareIter = (*name2Share).find(regName);
    if (shareIter != (*name2Share).end()) {
        // TODO: 需要根据Symtable结构正确处理
        return nullptr;
    }
    
    auto symIter = name2Sym.find(regName);
    if (symIter != name2Sym.end()) {
        // TODO: 需要根据Symtable结构正确处理
        return nullptr;
    }
    
    // 如果仍然没找到，创建一个新的寄存器
    PtxInterpreter::Reg* newReg = new PtxInterpreter::Reg();
    newReg->regType = Qualifier::S_UNKNOWN;  // 默认类型
    newReg->addr = malloc(8);  // 默认分配8字节
    memset(newReg->addr, 0, 8);
    name2Reg[regName] = newReg;
    return newReg->addr;
}

void* ThreadContext::get_memory_addr(OperandContext::FA *fa, std::vector<Qualifier> &qualifiers) {
    // 获取基础地址
    // 注意：这里需要根据FA结构的实际情况进行调整
    // 目前只是简单示例
    void* baseAddr = nullptr;
    
    if (fa->baseType == OperandContext::FA::CONSTANT) {
        auto iter = name2Sym.find(fa->baseName);
        if (iter != name2Sym.end()) {
            // TODO: 需要根据Symtable结构正确处理
        }
    } else if (fa->baseType == OperandContext::FA::SHARED) {
        auto iter = (*name2Share).find(fa->baseName);
        if (iter != (*name2Share).end()) {
            // TODO: 需要根据Symtable结构正确处理
        }
    }
    
    // 计算偏移量
    int offset = 0;
    if (fa->offsetType == OperandContext::FA::IMMEDIATE) {
        // 解析立即数偏移
        try {
            offset = std::stoi(fa->offsetVal);
        } catch (...) {
            offset = 0;
        }
    } else {
        // 寄存器偏移，需要获取寄存器的值
        if (fa->reg) {
            void* regAddr = get_operand_addr(*fa->reg, qualifiers);
            if (regAddr) {
                offset = *(int*)regAddr;
            }
        }
    }
    
    // 根据类型获取大小
    int size = TypeUtils::get_bytes(qualifiers);
    
    // 返回计算后的地址
    return (char*)baseAddr + offset * size;
}

void ThreadContext::mov_data(void *src, void *dst, std::vector<Qualifier> &qualifiers) {
    mov(src, dst, qualifiers);
}

bool ThreadContext::QvecHasQ(std::vector<Qualifier> &qvec, Qualifier q) {
    return std::find(qvec.begin(), qvec.end(), q) != qvec.end();
}

int ThreadContext::getBytes(std::vector<Qualifier> &q) {
    return TypeUtils::get_bytes(q);
}

void ThreadContext::mov(void *from, void *to, std::vector<Qualifier> &q) {
    int bytes = getBytes(q);
    if (bytes > 0) {
        memcpy(to, from, bytes);
    }
}

Qualifier ThreadContext::getDataType(std::vector<Qualifier> &q) {
    if (!q.empty()) {
        return q.back();
    }
    return Qualifier::S_UNKNOWN;
}

int ThreadContext::getBytes(Qualifier q) {
    return Q2bytes(q);
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

ThreadContext::DTYPE ThreadContext::getDType(std::vector<Qualifier> &q) {
    Qualifier type = getDataType(q);
    if (type == Qualifier::Q_F32 || type == Qualifier::Q_F64) 
        return DFLOAT;
    else 
        return DINT;
}

ThreadContext::DTYPE ThreadContext::getDType(Qualifier q) {
    if (q == Qualifier::Q_F32 || q == Qualifier::Q_F64) 
        return DFLOAT;
    else 
        return DINT;
}

bool ThreadContext::Signed(Qualifier q) {
    // TODO: 实现Signed函数
    return false;
}