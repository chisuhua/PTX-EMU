#include "ptxsim/thread_context.h"
#include "../utils/qualifier_utils.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/interpreter.h"
#include "ptxsim/ptx_debug.h"
#include <algorithm>
#include <any>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>

// 添加SHMEMADDR变量定义，用于处理shared memory地址
static uint64_t SHMEMADDR = 0;

#ifdef DEBUGINTE
extern bool sync_thread;
#endif
#ifdef LOGINTE
extern bool IFLOG();
#endif

void ThreadContext::init(
    Dim3 &blockIdx, Dim3 &threadIdx, Dim3 GridDim, Dim3 BlockDim,
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
    if (state != RUN)
        return state;

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

    // 使用DebugConfig获取完整的指令字符串（包含操作数）
    std::string operands =
        ptxsim::DebugConfig::get_full_instruction_string(statement);

    // 使用PTX_TRACE_INSTR宏跟踪指令执行
    PTX_TRACE_INSTR(pc, opcode, operands, BlockIdx, ThreadIdx);

    // 记录性能统计
    ptxsim::PTXDebugger::get().get_perf_stats().record_instruction(opcode);

    // 使用工厂创建对应的处理器并执行
    InstructionHandler *handler =
        InstructionFactory::create_handler(statement.statementType);
    if (handler) {
        handler->execute(this, statement);
        pc++;
    } else {
        std::cerr << "No handler found for statement type: "
                  << static_cast<int>(statement.statementType) << std::endl;
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

void ThreadContext::prepare_breakpoint_context(
    std::unordered_map<std::string, std::any> &context) {
    // 添加寄存器值
    for (const auto &reg_pair : name2Reg) {
        auto reg = reg_pair.second;
        if (reg && reg->addr) {
            switch (reg->regType) {
            case Qualifier::Q_U32:
            case Qualifier::Q_S32:
                context[reg_pair.first] = *(int32_t *)reg->addr;
                break;
            case Qualifier::Q_U64:
            case Qualifier::Q_S64:
                context[reg_pair.first] = *(int64_t *)reg->addr;
                break;
            case Qualifier::Q_F32:
                context[reg_pair.first] = *(float *)reg->addr;
                break;
            case Qualifier::Q_F64:
                context[reg_pair.first] = *(double *)reg->addr;
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

void ThreadContext::dump_state(std::ostream &os) const {
    // 寄存器状态
    os << "Registers:" << std::endl;
    for (const auto &reg_pair : name2Reg) {
        auto reg = reg_pair.second;
        if (reg && reg->addr) {
            os << "  " << reg_pair.first << " = ";
            switch (reg->regType) {
            case Qualifier::Q_U32:
                os << ptxsim::debug_format::format_u32(*(uint32_t *)reg->addr,
                                                       true);
                break;
            case Qualifier::Q_S32:
                os << ptxsim::debug_format::format_i32(*(int32_t *)reg->addr,
                                                       true);
                break;
            case Qualifier::Q_U64:
                os << ptxsim::debug_format::format_i64(*(uint64_t *)reg->addr,
                                                       true);
                break;
            case Qualifier::Q_S64:
                os << ptxsim::debug_format::format_i64(*(int64_t *)reg->addr,
                                                       true);
                break;
            case Qualifier::Q_F32:
                os << ptxsim::debug_format::format_f32(*(float *)reg->addr);
                break;
            case Qualifier::Q_F64:
                os << ptxsim::debug_format::format_f64(*(double *)reg->addr);
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

void *ThreadContext::get_operand_addr(OperandContext &op,
                                      std::vector<Qualifier> &qualifiers) {
    switch (op.operandType) {
    case O_REG:
        return get_register_addr((OperandContext::REG *)op.operand);

    case O_FA:
        return get_memory_addr((OperandContext::FA *)op.operand, qualifiers);

    case O_IMM: {
        auto immOp = (OperandContext::IMM *)op.operand;
        // 使用setIMM函数设置立即数
        int bytes = TypeUtils::get_bytes(qualifiers);
        Qualifier q;
        switch (bytes) {
        case 1:
            q = Qualifier::Q_U8;
            break;
        case 2:
            q = Qualifier::Q_U16;
            break;
        case 4:
            q = Qualifier::Q_U32;
            break;
        case 8:
            q = Qualifier::Q_U64;
            break;
        default:
            q = Qualifier::Q_U32;
            break;
        }
        setIMM(immOp->immVal, q);
        void *ret = &(imm.front()->data);
        imm.pop();
        return ret;
    }

    case O_VEC: {
        auto vecOp = (OperandContext::VEC *)op.operand;
        // 创建一个新的VEC对象用于存储向量元素地址
        PtxInterpreter::VEC *newVec = new PtxInterpreter::VEC();
        // 递归处理向量中的每个元素
        for (auto &elem : vecOp->vec) {
            newVec->vec.push_back(get_operand_addr(elem, qualifiers));
        }
        vec.push(newVec);
        return nullptr;
    }

    case O_VAR: {
        auto varOp = (OperandContext::VAR *)op.operand;
        // 查找共享内存中的变量
        auto share_it = name2Share->find(varOp->varName);
        if (share_it != name2Share->end()) {
            return &(share_it->second->val);
        }

        // 查找符号表中的变量
        auto sym_it = name2Sym.find(varOp->varName);
        if (sym_it != name2Sym.end()) {
            return &(sym_it->second->val);
        }

        // 如果都没找到，报错
        assert(0);
    }

    default:
        return nullptr;
    }
}

void *ThreadContext::get_register_addr(OperandContext::REG *reg,
                                       Qualifier qualifier) {
    // 首先检查是否为特殊寄存器（如%tid.x）
    if (reg->regName == "tid.x")
        return &ThreadIdx.x;
    if (reg->regName == "tid.y")
        return &ThreadIdx.y;
    if (reg->regName == "tid.z")
        return &ThreadIdx.z;
    if (reg->regName == "ctaid.x")
        return &BlockIdx.x;
    if (reg->regName == "ctaid.y")
        return &BlockIdx.y;
    if (reg->regName == "ctaid.z")
        return &BlockIdx.z;
    if (reg->regName == "nctaid.x")
        return &GridDim.x;
    if (reg->regName == "nctaid.y")
        return &GridDim.y;
    if (reg->regName == "nctaid.z")
        return &GridDim.z;
    if (reg->regName == "ntid.x")
        return &BlockDim.x;
    if (reg->regName == "ntid.y")
        return &BlockDim.y;
    if (reg->regName == "ntid.z")
        return &BlockDim.z;

    // 然后尝试直接按regName查找（适用于普通寄存器）
    auto it = name2Reg.find(reg->regName);
    if (it != name2Reg.end()) {
        return it->second->addr;
    }

    // 如果没找到，尝试组合名称查找（例如 regName="r", regIdx=1 组合成 "r1"）
    std::string combinedName = reg->regName + std::to_string(reg->regIdx);
    it = name2Reg.find(combinedName);
    if (it != name2Reg.end()) {
        return it->second->addr;
    }

    // 如果仍然找不到，创建新的寄存器
    PtxInterpreter::Reg *newReg = new PtxInterpreter::Reg();
    // 注意：这里需要从其他地方获取寄存器类型，因为OperandContext::REG没有保存类型信息
    // 我们暂时使用传入的qualifier参数或者默认类型
    newReg->regType = qualifier;

    // 根据类型分配数据空间 (创建一个只有一个元素的vector)
    std::vector<Qualifier> typeVec = {newReg->regType};
    int bytes = TypeUtils::get_bytes(typeVec);
    newReg->addr = malloc(bytes);
    memset(newReg->addr, 0, bytes);

    name2Reg[combinedName] = newReg;
    return newReg->addr;
}

void *ThreadContext::get_memory_addr(OperandContext::FA *fa,
                                     std::vector<Qualifier> &qualifiers) {
    void *ret;
    if (fa->reg) {
        // 1. 执行到get_memory_addr时，传入的qualifiers含有Q_GLOBAL, Q_SHARED,
        // Q_PARAM如何处理地址信息，
        // 把qualifiers里的地址信息设定到mem_qualifiers，而不是假定哟个Q_U64.
        Qualifier mem_qualifier;
        for (const auto &q : qualifiers) {
            // 将地址空间信息添加到mem_qualifiers中
            if (q == Qualifier::Q_GLOBAL || q == Qualifier::Q_SHARED ||
                q == Qualifier::Q_PARAM) {
                mem_qualifier = Qualifier::Q_U64;
            }
        }

        assert(fa->reg->operandType == O_REG);

        void *regAddr = get_register_addr(
            (OperandContext::REG *)(fa->reg)->operand, mem_qualifier);
        if (!regAddr)
            return nullptr;

        // 根据地址类型决定如何解读寄存器内容
        // 地址通常应该是64位的
        ret = (void *)*(uint64_t *)regAddr;
    } else {
        // 直接通过ID查找符号表或共享内存
        auto sym_it = name2Sym.find(fa->ID);
        if (sym_it != name2Sym.end()) {
            ret = (void *)sym_it->second->val;
        } else {
            auto share_it = name2Share->find(fa->ID);
            if (share_it != name2Share->end()) {
                // 处理shared memory地址
                extern uint64_t SHMEMADDR;
                ret = (void *)(share_it->second->val + (SHMEMADDR << 32));
            } else {
                assert(0);
            }
        }
    }

    // 如果是shared memory访问，需要特殊处理高32位地址
    // 这适用于通过寄存器访问shared memory的情况
    if (QvecHasQ(qualifiers, Qualifier::Q_SHARED)) {
        extern uint64_t SHMEMADDR;
        ret = (void *)((uint64_t)ret + (SHMEMADDR << 32));
    }

    // 处理偏移量
    if (!fa->offsetVal.empty()) {
        // 创建一个临时立即数操作数来处理偏移量
        OperandContext offsetOp;
        offsetOp.operandType = O_IMM;

        // 解析偏移量字符串为整数值
        OperandContext::IMM *immOffset = new OperandContext::IMM();
        try {
            immOffset->immVal = std::to_string(std::stoll(fa->offsetVal));
        } catch (...) {
            immOffset->immVal = "0"; // 默认偏移量为0
        }

        offsetOp.operand = immOffset;
        std::vector<Qualifier> offsetQualifiers = {Qualifier::Q_S64};
        void *offsetAddr = get_operand_addr(offsetOp, offsetQualifiers);

        if (offsetAddr) {
            int64_t offset = *(int64_t *)offsetAddr;
            ret = (void *)((uint64_t)ret + offset);
        }

        // 注意：这里不应该手动删除immOffset，因为offsetOp析构时会自动删除
        // delete immOffset;
    }

    return ret;
}

void ThreadContext::mov_data(void *src, void *dst,
                             std::vector<Qualifier> &qualifiers) {
    int bytes = TypeUtils::get_bytes(qualifiers);
    memcpy(dst, src, bytes);
}

void ThreadContext::handle_statement(StatementContext &statement) {
    // 使用工厂创建对应的处理器并执行
    InstructionHandler *handler =
        InstructionFactory::create_handler(statement.statementType);
    if (handler) {
        handler->execute(this, statement);
    } else {
        std::cerr << "No handler found for statement type: "
                  << static_cast<int>(statement.statementType) << std::endl;
    }
}

// 添加shared memory初始化函数
void ThreadContext::initialize_shared_memory(const std::string &name,
                                             uint64_t address) {
    extern uint64_t SHMEMADDR;
    if (SHMEMADDR) {
        assert(address >> 32 == SHMEMADDR);
    } else {
        SHMEMADDR = address >> 32; // 只保存高32位
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

void ThreadContext::trace_register(OperandContext::REG *reg, void *value,
                                   std::vector<Qualifier> &qualifiers,
                                   bool is_write) {
    // 检查寄存器是否有效
    if (!reg)
        return;

    // 获取完整的寄存器名称
    std::string reg_name = reg->getFullName();

    // 获取更新后的值用于跟踪（从传入的value参数中）
    int bytes = TypeUtils::get_bytes(qualifiers);
    std::any reg_value;
    switch (bytes) {
    case 1:
        reg_value = *(uint8_t *)value;
        break;
    case 2:
        reg_value = *(uint16_t *)value;
        break;
    case 4:
        if (TypeUtils::is_float_type(qualifiers)) {
            reg_value = *(float *)value;
        } else {
            reg_value = *(uint32_t *)value;
        }
        break;
    case 8:
        if (TypeUtils::is_float_type(qualifiers)) {
            reg_value = *(double *)value;
        } else {
            reg_value = *(uint64_t *)value;
        }
        break;
    }

    PTX_TRACE_REG_ACCESS(reg_name, reg_value, is_write, BlockIdx, ThreadIdx);
}

void ThreadContext::memory_access(bool is_write, const std::string &addr_expr,
                                  void *addr, size_t size, void *value,
                                  std::vector<Qualifier> &qualifiers,
                                  void *target,
                                  OperandContext::REG *reg_operand) {
    // 只有在启用了内存跟踪时才进行跟踪
    if (ptxsim::DebugConfig::get().is_memory_traced()) {
        // 获取数据值用于跟踪
        void *track_value = nullptr;

        if (is_write) {
            // 写操作：追踪要被写入内存的源数据
            track_value = value;
        } else {
            // 读操作：追踪即将从内存中读取的数据
            track_value = addr;
        }

        PTX_TRACE_MEM_ACCESS(is_write, addr_expr, (uint64_t)addr, size,
                             track_value, BlockIdx, ThreadIdx);
    }

    // 执行实际的内存操作
    if (is_write) {
        // 将源数据 'value' 复制到内存地址 'addr'
        mov(value, addr, qualifiers);
    } else {
        // 将内存地址 'addr' 中的数据复制到目标 'target'
        mov(addr, target, qualifiers);
    }

    // 根据操作类型跟踪寄存器访问
    if (reg_operand) {
        if (is_write) {
            // 内存写操作：源寄存器（包含要写的数据）被读取
            trace_register(reg_operand, value, qualifiers, false);
        } else {
            // 内存读操作：目标寄存器（将接收数据）被写入
            trace_register(reg_operand, target, qualifiers, true);
        }
    }
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
    PtxInterpreter::IMM *immObj = new PtxInterpreter::IMM();

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
    PtxInterpreter::IMM *immObj = new PtxInterpreter::IMM();
    immObj->type = q;

    // 根据限定符类型设置值，与原始实现保持一致
    switch (q) {
    case Qualifier::Q_S64:
    case Qualifier::Q_U64:
    case Qualifier::Q_B64:
        immObj->data.u64 = std::stoll(s, 0, 0);
        break;
    case Qualifier::Q_S32:
    case Qualifier::Q_U32:
    case Qualifier::Q_B32:
        immObj->data.u32 = std::stoi(s, 0, 0);
        break;
    case Qualifier::Q_S16:
    case Qualifier::Q_U16:
    case Qualifier::Q_B16:
        immObj->data.u16 = (uint16_t)std::stoi(s, 0, 0);
        break;
    case Qualifier::Q_S8:
    case Qualifier::Q_U8:
    case Qualifier::Q_B8:
    case Qualifier::Q_PRED:
        immObj->data.u8 = (uint8_t)std::stoi(s, 0, 0);
        break;
    case Qualifier::Q_F64:
        // 处理双精度浮点数
        if (s.size() == 18 && (s[1] == 'd' || s[1] == 'D')) {
            s[1] = 'x';
            *(uint64_t *)&(immObj->data.f64) = std::stoull(s, 0, 0);
        } else {
            immObj->data.f64 = std::stod(s);
        }
        break;
    case Qualifier::Q_F32:
        // 处理单精度浮点数
        if (s.size() == 10 && (s[1] == 'f' || s[1] == 'F')) {
            s[1] = 'x';
            // 当使用stoi处理输入0xBF000000时会抛出std::out_of_range异常
            *(uint32_t *)&(immObj->data.f32) = (uint32_t)std::stol(s, 0, 0);
        } else {
            immObj->data.f32 = std::stof(s);
        }
        break;
    default:
        assert(0);
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
    if (q.empty())
        return DNONE;
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