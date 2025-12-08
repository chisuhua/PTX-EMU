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
    assert(state == RUN);
    assert(pc >= 0 && pc < statements->size());
    
    // 准备断点检查上下文
    std::unordered_map<std::string, std::any> context;
    prepare_breakpoint_context(context);
    
    // 检查断点
    if (PTX_CHECK_BREAKPOINT(pc, context)) {
        state = (EXE_STATE)2; // BREAK状态
        PTX_DUMP_THREAD_STATE("Breakpoint hit", *this, BlockIdx, ThreadIdx);
        return; // 暂停执行
    }
    
    // 开始性能计时
    PTX_PERF_TIMER("instruction_execution");
    
    // 跟踪指令
    StatementContext &statement = (*statements)[pc];
    std::string opcode = S2s(statement.statementType);
    
    // 构建操作数字符串
    std::string operands = "";
    switch (statement.statementType) {
        case StatementType::S_MOV: {
            auto* mov_stmt = static_cast<StatementContext::MOV*>(statement.statement);
            if (mov_stmt) {
                operands = mov_stmt->movOp[0].toString() + ", " + mov_stmt->movOp[1].toString();
            }
            break;
        }
        case StatementType::S_ADD: {
            auto* add_stmt = static_cast<StatementContext::ADD*>(statement.statement);
            if (add_stmt) {
                operands = add_stmt->addOp[0].toString() + ", " + add_stmt->addOp[1].toString() 
                          + ", " + add_stmt->addOp[2].toString();
            }
            break;
        }
        case StatementType::S_ST: {
            auto* st_stmt = static_cast<StatementContext::ST*>(statement.statement);
            if (st_stmt) {
                operands = st_stmt->stOp[0].toString() + ", " + st_stmt->stOp[1].toString();
            }
            break;
        }
        case StatementType::S_LD: {
            auto* ld_stmt = static_cast<StatementContext::LD*>(statement.statement);
            if (ld_stmt) {
                operands = ld_stmt->ldOp[0].toString() + ", " + ld_stmt->ldOp[1].toString();
            }
            break;
        }
        case StatementType::S_SETP: {
            auto* setp_stmt = static_cast<StatementContext::SETP*>(statement.statement);
            if (setp_stmt) {
                operands = setp_stmt->setpOp[0].toString() + ", " + setp_stmt->setpOp[1].toString() 
                          + ", " + setp_stmt->setpOp[2].toString();
            }
            break;
        }
        case StatementType::S_MUL: {
            auto* mul_stmt = static_cast<StatementContext::MUL*>(statement.statement);
            if (mul_stmt) {
                operands = mul_stmt->mulOp[0].toString() + ", " + mul_stmt->mulOp[1].toString() 
                          + ", " + mul_stmt->mulOp[2].toString();
            }
            break;
        }
        default:
            // 对于其他指令类型，暂时使用占位符
            operands = "...";
            break;
    }
    
    PTX_TRACE_INSTR(pc, opcode.c_str(), operands.c_str());
    
    // 记录性能统计
    ptxsim::PTXDebugger::get().get_perf_stats().record_instruction(opcode);
    
    // 使用工厂创建对应的处理器并执行
    InstructionHandler* handler = InstructionFactory::create_handler(statement.statementType);
    if (handler) {
        handler->execute(this, statement);
        pc++;
    } else {
        std::cerr << "No handler found for statement type: " << static_cast<int>(statement.statementType) << std::endl;
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

void ThreadContext::prepare_breakpoint_context(std::unordered_map<std::string, std::any>& context) {
    // 添加寄存器值
    for (const auto& reg_pair : name2Reg) {
        auto reg = reg_pair.second;
        if (reg && reg->addr) {
            switch (reg->regType) {
                case Qualifier::Q_U32:
                case Qualifier::Q_S32:
                    context[reg_pair.first] = *(int32_t*)reg->addr;
                    break;
                case Qualifier::Q_U64:
                case Qualifier::Q_S64:
                    context[reg_pair.first] = *(int64_t*)reg->addr;
                    break;
                case Qualifier::Q_F32:
                    context[reg_pair.first] = *(float*)reg->addr;
                    break;
                case Qualifier::Q_F64:
                    context[reg_pair.first] = *(double*)reg->addr;
                    break;
                default:
                    break;
            }
        }
    }
    
    // 添加特殊寄存器
    context["pc"] = pc;
    context["tid.x"] = (int)ThreadIdx.x;
    context["tid.y"] = (int)ThreadIdx.y;
    context["tid.z"] = (int)ThreadIdx.z;
    context["bid.x"] = (int)BlockIdx.x;
    context["bid.y"] = (int)BlockIdx.y;
    context["bid.z"] = (int)BlockIdx.z;
}

void ThreadContext::dump_state(std::ostream& os) const {
    // 寄存器状态
    os << "Registers:" << std::endl;
    for (const auto& reg_pair : name2Reg) {
        auto reg = reg_pair.second;
        if (reg && reg->addr) {
            os << "  " << reg_pair.first << " = ";
            switch (reg->regType) {
                case Qualifier::Q_U32:
                    os << ptxsim::debug_format::format_u32(*(uint32_t*)reg->addr, true);
                    break;
                case Qualifier::Q_S32:
                    os << ptxsim::debug_format::format_i32(*(int32_t*)reg->addr, true);
                    break;
                case Qualifier::Q_U64:
                    os << ptxsim::debug_format::format_i64(*(uint64_t*)reg->addr, true);
                    break;
                case Qualifier::Q_S64:
                    os << ptxsim::debug_format::format_i64(*(int64_t*)reg->addr, true);
                    break;
                case Qualifier::Q_F32:
                    os << ptxsim::debug_format::format_f32(*(float*)reg->addr);
                    break;
                case Qualifier::Q_F64:
                    os << ptxsim::debug_format::format_f64(*(double*)reg->addr);
                    break;
                default:
                    os << "[unsupported type]";
                    break;
            }
            os << std::endl;
        }
    }
    
    // 当前PC和指令
    if (pc >= 0 && pc < (int)statements->size()) {
        StatementContext &statement = (*statements)[pc];
        os << "Current Instruction:" << std::endl;
        os << "  [" << pc << "] " << S2s(statement.statementType) << std::endl;
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
            return nullptr;
    }
}

void* ThreadContext::get_register_addr(OperandContext::REG *reg) {
    std::string regName = reg->regName;
    
    // 检查寄存器是否已存在
    auto it = name2Reg.find(regName);
    if (it != name2Reg.end()) {
        return it->second->addr;
    }
    
    // 如果不存在，创建新的寄存器
    PtxInterpreter::Reg* newReg = new PtxInterpreter::Reg();
    // 注意：这里需要从其他地方获取寄存器类型，因为OperandContext::REG没有保存类型信息
    // 我们暂时使用默认类型，后续可以通过上下文获取正确的类型
    newReg->regType = Qualifier::Q_U32;
    
    // 根据类型分配数据空间 (创建一个只有一个元素的vector)
    std::vector<Qualifier> typeVec = {newReg->regType};
    int bytes = TypeUtils::get_bytes(typeVec);
    newReg->addr = malloc(bytes);
    memset(newReg->addr, 0, bytes);
    
    name2Reg[regName] = newReg;
    return newReg->addr;
}

void* ThreadContext::get_memory_addr(OperandContext::FA *fa, std::vector<Qualifier> &qualifiers) {
    // 获取基础地址
    void* baseAddr = get_operand_addr(*(fa->reg), qualifiers);
    if (!baseAddr) return nullptr;
    
    // 计算偏移量 (注意：这里应该解析offsetVal，但我们暂时使用0)
    int offset = 0;
    
    // 返回最终地址
    return (char*)baseAddr + offset;
}

void ThreadContext::mov_data(void *src, void *dst, std::vector<Qualifier> &qualifiers) {
    int bytes = TypeUtils::get_bytes(qualifiers);
    memcpy(dst, src, bytes);
}

void ThreadContext::handle_statement(StatementContext &statement) {
    // 使用工厂创建对应的处理器并执行
    InstructionHandler* handler = InstructionFactory::create_handler(statement.statementType);
    if (handler) {
        handler->execute(this, statement);
    } else {
        std::cerr << "No handler found for statement type: " << static_cast<int>(statement.statementType) << std::endl;
    }
}

bool ThreadContext::QvecHasQ(std::vector<Qualifier> &qvec, Qualifier q) {
    return std::find(qvec.begin(), qvec.end(), q) != qvec.end();
}

int ThreadContext::getBytes(std::vector<Qualifier> &q) {
    return TypeUtils::get_bytes(q);
}

void ThreadContext::mov(void *from, void *to, std::vector<Qualifier> &q) {
    int bytes = TypeUtils::get_bytes(q);
    memcpy(to, from, bytes);
}

bool ThreadContext::isIMMorVEC(OperandContext &op) {
    return (op.operandType == O_IMM || op.operandType == O_VEC);
}

Qualifier ThreadContext::getCMPOP(std::vector<Qualifier> &q) {
    return TypeUtils::get_comparison_op(q);
}

bool ThreadContext::is_immediate_or_vector(OperandContext &op) {
    return (op.operandType == O_IMM || op.operandType == O_VEC);
}

void ThreadContext::set_immediate_value(std::string value, Qualifier type) {
    // 创建一个新的IMM对象
    PtxInterpreter::IMM* immObj = new PtxInterpreter::IMM();
    
    // 根据类型设置值
    switch (type) {
        case Qualifier::Q_U32:
        case Qualifier::Q_S32:
            immObj->data.u32 = std::stoi(value);
            break;
        case Qualifier::Q_U64:
        case Qualifier::Q_S64:
            immObj->data.u64 = std::stoll(value);
            break;
        case Qualifier::Q_F32:
            immObj->data.f32 = std::stof(value);
            break;
        case Qualifier::Q_F64:
            immObj->data.f64 = std::stod(value);
            break;
        default:
            // 默认情况下，尝试作为整数处理
            immObj->data.u32 = std::stoi(value);
            break;
    }
    
    // 将IMM对象加入队列
    imm.push(immObj);
}

Qualifier ThreadContext::getDataType(std::vector<Qualifier> &q) {
    // 返回向量中的第一个限定符作为数据类型
    if (!q.empty()) {
        return q[0];
    }
    return Qualifier::S_UNKNOWN;
}

int ThreadContext::getBytes(Qualifier q) {
    std::vector<Qualifier> typeVec = {q};
    return TypeUtils::get_bytes(typeVec);
}

void ThreadContext::setIMM(std::string s, Qualifier q) {
    // 创建一个新的IMM对象
    PtxInterpreter::IMM* immObj = new PtxInterpreter::IMM();
    
    // 根据限定符类型设置值
    switch (q) {
        case Qualifier::Q_U32:
        case Qualifier::Q_S32:
            immObj->data.u32 = std::stoi(s);
            break;
        case Qualifier::Q_U64:
        case Qualifier::Q_S64:
            immObj->data.u64 = std::stoll(s);
            break;
        case Qualifier::Q_F32:
            immObj->data.f32 = std::stof(s);
            break;
        case Qualifier::Q_F64:
            immObj->data.f64 = std::stod(s);
            break;
        default:
            // 默认情况下，尝试作为整数处理
            immObj->data.u32 = std::stoi(s);
            break;
    }
    
    // 将IMM对象加入队列
    imm.push(immObj);
}

void ThreadContext::clearIMM_VEC() {
    // 清理IMM队列
    while (!imm.empty()) {
        delete imm.front();
        imm.pop();
    }
    
    // 清理VEC队列
    while (!vec.empty()) {
        delete vec.front();
        vec.pop();
    }
}

ThreadContext::DTYPE ThreadContext::getDType(std::vector<Qualifier> &q) {
    if (q.empty()) return DNONE;
    return getDType(q[0]);
}

ThreadContext::DTYPE ThreadContext::getDType(Qualifier q) {
    std::vector<Qualifier> typeVec = {q};
    if (TypeUtils::is_float_type(typeVec)) {
        return DFLOAT;
    } else if (q != Qualifier::S_UNKNOWN) {
        return DINT;
    }
    return DNONE;
}

bool ThreadContext::Signed(Qualifier q) {
    // 判断是否是有符号类型
    switch (q) {
        case Qualifier::Q_S8:
        case Qualifier::Q_S16:
        case Qualifier::Q_S32:
        case Qualifier::Q_S64:
            return true;
        default:
            return false;
    }
}