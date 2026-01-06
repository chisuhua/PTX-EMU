#include "ptxsim/thread_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/interpreter.h"
#include "ptxsim/ptx_debug.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "utils/logger.h"
#include <algorithm>
#include <any>
#include <cassert>
#include <cmath>
#include <cstdint> // 添加此行以支持uint64_t
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

void ThreadContext::init(Dim3 &blockIdx, Dim3 &threadIdx, Dim3 GridDim,
                         Dim3 BlockDim,
                         std::vector<StatementContext> &statements,
                         std::map<std::string, Symtable *> &name2Share,
                         std::map<std::string, Symtable *> &name2Sym,
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
    this->next_pc = 0;
    this->state = RUN;
    operand_collected.resize(4); // max operands perf instruction reserved is 4

    // 计算并设置warp_id和lane_id
    int thread_id = ThreadIdx.x + ThreadIdx.y * BlockDim.x +
                    ThreadIdx.z * BlockDim.x * BlockDim.y;
    this->warp_id_ = thread_id / WarpContext::WARP_SIZE;
    this->lane_id_ = thread_id % WarpContext::WARP_SIZE;

    // 注意：寄存器管理现在完全由RegisterBankManager负责
    // 寄存器预分配现在由CTAContext统一处理
}

void ThreadContext::_execute_once() {
    assert(state == RUN);
    // 使用安全的PC检查
    assert(is_valid_pc());

    // 准备断点检查上下文
    // std::unordered_map<std::string, std::any> context;
    // prepare_breakpoint_context(context);

    // // 检查断点
    // if (PTX_CHECK_BREAKPOINT(pc, context)) {
    //     state = (EXE_STATE)2; // BREAK状态
    //     PTX_DUMP_THREAD_STATE("Breakpoint hit", *this, BlockIdx, ThreadIdx);
    //     return; // 暂停执行
    // }

    // 开始性能计时
    // PTX_PERF_TIMER("instruction_execution");

    // 跟踪指令
    StatementContext &statement = (*statements)[pc];

    if (statement.state == InstructionState::READY) {
        trace_instruction(statement);
    }

    // 使用工厂创建对应的处理器
    InstructionHandler *handler =
        InstructionFactory::get_handler(statement.statementType);
    if (handler) {
        // 直接调用execute_full方法执行整个指令
        handler->execute(this, statement);
    } else {
        std::cerr << "No handler found for statement type: "
                  << static_cast<int>(statement.statementType) << std::endl;
        state = EXIT;
    }

    // 更新PC
    pc = next_pc;
}

void ThreadContext::trace_instruction(StatementContext &statement) {
    std::string opcode = S2s(statement.statementType);

    // 使用DebugConfig获取完整的指令字符串（包含操作数）
    std::string operands =
        ptxsim::DebugConfig::get_full_instruction_string(statement);

    // 使用PTX_TRACE_INSTR宏跟踪指令执行
    PTX_TRACE_INSTR(pc, opcode, operands, BlockIdx, ThreadIdx);

    // 记录性能统计
    ptxsim::PTXDebugger::get().get_perf_stats().record_instruction(opcode);
}

void ThreadContext::clear_temporaries() {
    while (!vec.empty()) {
        delete vec.front();
        vec.pop();
    }
}

void ThreadContext::prepare_breakpoint_context(
    std::unordered_map<std::string, std::any> &context) {
    // 使用RegisterBankManager获取寄存器值
    if (register_bank_manager_) {
        // 计算warp_id和lane_id
        int warp_id = (ThreadIdx.x + ThreadIdx.y * BlockDim.x +
                       ThreadIdx.z * BlockDim.x * BlockDim.y) /
                      WarpContext::WARP_SIZE;
        int lane_id = (ThreadIdx.x + ThreadIdx.y * BlockDim.x +
                       ThreadIdx.z * BlockDim.x * BlockDim.y) %
                      WarpContext::WARP_SIZE;

        // TODO
        // // 遍历所有寄存器获取值
        // auto all_registers = register_bank_manager_->get_all_registers();
        // for (const auto &reg_name : all_registers) {
        //     void *reg_data = register_bank_manager_->get_register(
        //         reg_name, warp_id, lane_id);
        //     if (reg_data) {
        //         // 根据寄存器大小推测类型
        //         //
        //         这里假设寄存器大小不超过8字节，实际大小需要从RegisterBankManager获取
        //         size_t reg_size = 8; // TODO: 获取实际寄存器大小
        //         uint64_t val = 0;
        //         memcpy(&val, reg_data, std::min(sizeof(val), reg_size));
        //         context[reg_name] = val;
        //     }
        // }
    }

    // 添加其他上下文信息
    context["pc"] = pc;
    context["blockIdx"] = BlockIdx;
    context["threadIdx"] = ThreadIdx;
}

void ThreadContext::dump_state(std::ostream &os) const {
    os << "Thread State:" << std::endl;
    os << "  BlockIdx: [" << BlockIdx.x << ", " << BlockIdx.y << ", "
       << BlockIdx.z << "]" << std::endl;
    os << "  ThreadIdx: [" << ThreadIdx.x << ", " << ThreadIdx.y << ", "
       << ThreadIdx.z << "]" << std::endl;
    os << "  PC: " << pc << std::endl;
    os << "  State: ";
    switch (state) {
    case RUN:
        os << "RUN";
        break;
    case EXIT:
        os << "EXIT";
        break;
    case BAR_SYNC:
        os << "BAR_SYNC";
        break;
    default:
        os << "UNKNOWN";
        break;
    }
    os << std::endl;

    os << "  Condition Codes: ";
    os << "carry=" << cc_reg.get_carry() << ", ";
    os << "overflow=" << cc_reg.get_overflow() << ", ";
    os << "zero=" << cc_reg.get_zero() << ", ";
    os << "sign=" << cc_reg.get_sign() << std::endl;
}

void ThreadContext::reset() {
    pc = 0;
    next_pc = 0;
    state = RUN;
    cc_reg = ConditionCodeRegister{}; // 重置条件码寄存器

    // 清空临时数据（寄存器管理由RegisterBankManager负责，无需本地重置）
    clear_temporaries();
    operand_collected.clear();
    operand_collected.resize(4);
}

// 添加新的执行方法
EXE_STATE ThreadContext::execute_thread_instruction() {
    this->_execute_once();
    return this->state; // 返回线程的实际状态
}

void *ThreadContext::acquire_operand(OperandContext &operand,
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
        return acquire_register((OperandContext::REG *)operand.operand,
                                qualifiers);

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
        VEC *newVec = new VEC();
        // 递归处理向量中的每个元素
        for (auto &elem : vecOp->vec) {
            newVec->vec.push_back(acquire_operand(elem, qualifiers));
        }
        vec.push(newVec);
        return nullptr;
    }

    default:
        break;
    }

    return nullptr;
}

void ThreadContext::collect_operands(StatementContext &stmt,
                                     std::vector<OperandContext> &operands,
                                     std::vector<Qualifier> *qualifier) {
    int bytes = getBytes(*qualifier);
    for (int i = 0; i < operands.size(); i++) {
        PTX_DEBUG_EMU("Collect: %s ", operands[i].toString(bytes).c_str());
        operand_collected[i] = operands[i].operand_phy_addr;
    }
    stmt.qualifier = qualifier;
};

void ThreadContext::commit_operand(StatementContext &stmt,
                                   OperandContext &operand,
                                   std::vector<Qualifier> &qualifier) {
    int bytes = getBytes(qualifier);
    PTX_DEBUG_EMU("Commit:  %s ", operand.toString(bytes).c_str());
};

void *ThreadContext::acquire_register(OperandContext::REG *reg,
                                      std::vector<Qualifier> qualifier) {
    // 确保register_bank_manager_存在
    if (!register_bank_manager_) {
        throw std::runtime_error("RegisterBankManager is required but not set");
    }

    // 计算warp_id和lane_id
    int warp_id = (ThreadIdx.x + ThreadIdx.y * BlockDim.x +
                   ThreadIdx.z * BlockDim.x * BlockDim.y) /
                  WarpContext::WARP_SIZE;
    int lane_id = (ThreadIdx.x + ThreadIdx.y * BlockDim.x +
                   ThreadIdx.z * BlockDim.x * BlockDim.y) %
                  WarpContext::WARP_SIZE;

    std::string combinedName = reg->regName + std::to_string(reg->regIdx);
    void *reg_data =
        register_bank_manager_->get_register(combinedName, warp_id, lane_id);

    // 如果寄存器不存在，直接断言退出
    assert(reg_data != nullptr &&
           "Register not found in bank manager. Aborting.");

    return reg_data;
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

        void *regAddr = acquire_register(
            (OperandContext::REG *)(fa->reg)->operand, {mem_qualifier});
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
                          "symbol_table_entry=%p, stored_value=0x%lx",
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