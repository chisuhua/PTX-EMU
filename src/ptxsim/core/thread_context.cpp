#include "ptxsim/thread_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/interpreter.h"
#include "ptxsim/ptx_debug.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "utils/logger.h"
#include <algorithm>
#include <any>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <queue>

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
    this->instruction_state = InstructionExecutionState::READY;

    // 重新初始化RegisterManager（清空所有寄存器）
    register_manager = RegisterManager();
}

EXE_STATE ThreadContext::exe_once() {
    if (state != RUN)
        return state;

    // 推进寄存器状态
    register_manager.tick();

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

    // 使用工厂创建对应的处理器
    InstructionHandler *handler =
        InstructionFactory::create_handler(statement.statementType);
    if (handler) {
        // 直接调用execute_full方法执行整个指令
        handler->execute_full(this, statement);
    } else {
        std::cerr << "No handler found for statement type: "
                  << static_cast<int>(statement.statementType) << std::endl;
        state = EXIT;
    }

    // 指令执行完成后，直接推进PC
    pc++;
}

void ThreadContext::clear_temporaries() {
    while (!vec.empty()) {
        delete vec.front();
        vec.pop();
    }
}

void ThreadContext::prepare_breakpoint_context(
    std::unordered_map<std::string, std::any> &context) {
    // 添加寄存器值 - 现在从RegisterManager获取
    for (const auto &reg_pair : register_manager.get_all_registers()) {
        std::string reg_name = reg_pair.first;
        RegisterInterface *reg_interface = reg_pair.second;
        if (reg_interface && reg_interface->get_physical_address()) {
            // 根据寄存器大小推测类型
            size_t reg_size = reg_interface->get_size();
            void *addr = reg_interface->get_physical_address();

            if (reg_size == 4) {
                context[reg_name] = *(int32_t *)addr;
            } else if (reg_size == 8) {
                context[reg_name] = *(int64_t *)addr;
            } else {
                // 其他大小的寄存器暂时跳过
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
    // 寄存器状态 - 现在从RegisterManager获取
    os << "Registers:" << std::endl;
    for (const auto &reg_pair : register_manager.get_all_registers()) {
        std::string reg_name = reg_pair.first;
        RegisterInterface *reg_interface = reg_pair.second;
        if (reg_interface && reg_interface->get_physical_address()) {
            os << "  " << reg_name << " = ";
            size_t reg_size = reg_interface->get_size();

            // 由于我们不知道寄存器的确切类型，需要基于大小进行推测
            if (reg_size == 4) {
                os << ptxsim::debug_format::format_i32(
                    *(int32_t *)reg_interface->get_physical_address(), true);
            } else if (reg_size == 8) {
                os << ptxsim::debug_format::format_i64(
                    *(int64_t *)reg_interface->get_physical_address(), true);
            } else {
                os << "[unknown size: " << reg_size << "]";
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

void *ThreadContext::get_operand_addr(OperandContext &operand,
                                      std::vector<Qualifier> &qualifiers) {
    switch (operand.operandType) {
    case O_VAR: {
        OperandContext::VAR *varOp = (OperandContext::VAR *)operand.operand;
        // 查找共享内存中的变量
        auto share_it = name2Share->find(varOp->varName);
        if (share_it != name2Share->end()) {
            return &(share_it->second->val);
        }

        auto sym_it = name2Sym.find(varOp->varName);
        if (sym_it != name2Sym.end()) {
            PTX_DEBUG_EMU("Reading kernel argument from name2Sym: name=%s, "
                          "symbol_table_entry=%p, stored_value=0x%lx, "
                          "dereferenced_value=0x%lx",
                          varOp->varName.c_str(), sym_it->second,
                          sym_it->second->val,
                          *(uint64_t *)(sym_it->second->val));
            return &(sym_it->second->val);
        }
        break;
    }

    case O_REG:
        return get_register_addr((OperandContext::REG *)operand.operand,
                                 getDataQualifier(qualifiers));

    case O_FA:
        return get_memory_addr((OperandContext::FA *)operand.operand,
                               qualifiers);

    case O_IMM: {
        auto immOp = (OperandContext::IMM *)operand.operand;
        Qualifier q = getDataQualifier(qualifiers);

        // 使用栈上缓冲区（每个 IMM 使用独立空间，支持多 IMM 指令）
        // 注意：此指针仅在当前指令执行期间有效！
        alignas(8) static thread_local char
            imm_buffer_pool[64][8]; // 支持最多 64 个 IMM/指令
        static thread_local int buffer_index = 0;

        // 使用模运算维护索引，避免溢出
        char *buffer = imm_buffer_pool[buffer_index];
        buffer_index = (buffer_index + 1) % 64;

        parseImmediate(immOp->immVal, q, buffer);
        return buffer;
    }

    case O_VEC: {
        auto vecOp = (OperandContext::VEC *)operand.operand;
        // 创建一个新的VEC对象用于存储向量元素地址
        PtxInterpreter::VEC *newVec = new PtxInterpreter::VEC();
        // 递归处理向量中的每个元素
        for (auto &elem : vecOp->vec) {
            newVec->vec.push_back(get_operand_addr(elem, qualifiers));
        }
        vec.push(newVec);
        return nullptr;
    }

    default:
        break;
    }

    return nullptr;
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
    std::string combinedName = reg->regName + std::to_string(reg->regIdx);

    // 检查寄存器是否已存在于RegisterManager中
    RegisterInterface *reg_interface =
        register_manager.get_register(combinedName);
    if (reg_interface) {
        return reg_interface->get_physical_address();
    }

    // 如果仍然找不到，创建新的寄存器
    // 根据类型分配数据空间
    std::vector<Qualifier> typeVec = {qualifier};
    int bytes = getBytes(typeVec);

    // 使用RegisterManager创建寄存器
    if (register_manager.create_register(combinedName, bytes)) {
        RegisterInterface *new_reg_interface =
            register_manager.get_register(combinedName);
        if (new_reg_interface) {
            // 初始化寄存器内容为0
            memset(new_reg_interface->get_physical_address(), 0, bytes);
            return new_reg_interface->get_physical_address();
        }
    }

    // 如果创建失败，返回nullptr
    return nullptr;
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
            PTX_DEBUG_EMU("Reading kernel argument from name2Sym in "
                          "get_memory_addr: name=%s, "
                          "symbol_table_entry=%p, stored_value=0x%lx, ",
                          fa->ID.c_str(), sym_it->second, sym_it->second->val);
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
        // 直接解析偏移量字符串，避免创建临时立即数操作数
        int64_t offset = 0;
        try {
            // 解析偏移量字符串为整数值
            offset = std::stoll(fa->offsetVal);
        } catch (...) {
            offset = 0; // 默认偏移量为0
        }

        ret = (void *)((uint64_t)ret + offset);
    }

    return ret;
}

void ThreadContext::mov_data(void *src, void *dst,
                             std::vector<Qualifier> &qualifiers) {
    int bytes = getBytes(qualifiers);
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

void ThreadContext::mov(void *from, void *to, const std::vector<Qualifier> &q) {
    int bytes = getBytes(q);
    memcpy(to, from, bytes);
}

bool ThreadContext::isIMMorVEC(OperandContext &op) {
    return (op.operandType == O_IMM || op.operandType == O_VEC);
}

bool ThreadContext::is_immediate_or_vector(OperandContext &op) {
    return (op.operandType == O_IMM || op.operandType == O_VEC);
}