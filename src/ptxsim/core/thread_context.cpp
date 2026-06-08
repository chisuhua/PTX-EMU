#include "ptxsim/thread_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/ptx_syntax_utils.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/ptx_debug.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/warp_context.h"
#include "utils/logger.h"
#include <algorithm>
#include <any>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <queue>

// 添加 SHMEMADDR 变量定义，用于处理 shared memory 地址
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
                         std::map<std::string, Symtable *> *name2Sym,
                         std::map<std::string, int> &label2pc,
                         std::map<std::string, Symtable *> *name2Share,
                         CTAContext *cta_ctx) {
    this->BlockIdx = blockIdx;
    this->ThreadIdx = threadIdx;
    this->GridDim = GridDim;
    this->BlockDim = BlockDim;
    this->statements = &statements;
    this->name2Sym = name2Sym;
    this->name2Share = name2Share; // 设置共享内存符号表引用
    this->label2pc = label2pc;
    this->bar_id = 0;  // Initialize barrier ID to default
    this->state = RUN;
    operand_collected.resize(ThreadContext::MAX_OPERANDS_PER_INSTR);
    operand_is_immediate_.resize(ThreadContext::MAX_OPERANDS_PER_INSTR);

    // 计算并设置warp_id和lane_id
    int thread_id = ThreadIdx.x + ThreadIdx.y * BlockDim.x +
                    ThreadIdx.z * BlockDim.x * BlockDim.y;
    this->warp_id_ = thread_id / WarpContext::WARP_SIZE;
    this->lane_id_ = thread_id % WarpContext::WARP_SIZE;

    // 设置CTAContext指针，用于访问本地内存符号表
    this->cta_context_ = cta_ctx;

    // 注意：寄存器管理现在完全由RegisterBankManager负责
    // 寄存器预分配现在由CTAContext统一处理
}

// 设置本地内存空间的方法实现
void ThreadContext::set_local_memory_space(void *local_mem_space) {
    this->local_mem_space = local_mem_space;
    PTX_DEBUG_EMU("Thread (%d,%d,%d) local memory space set to %p", ThreadIdx.x,
                  ThreadIdx.y, ThreadIdx.z, local_mem_space);
}

void ThreadContext::_execute_once() {
    // assert(state == RUN); // Allow BAR_SYNC threads to retry barrier
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
    int current_pc = get_pc();
    StatementContext &statement = (*statements)[current_pc];

    set_next_pc(current_pc + 1);

    // if (statement.state == InstructionState::READY) {
    //     trace_instruction(statement);
    // }

    // 使用工厂创建对应的处理器
    InstructionHandler *handler =
        InstructionFactory::get_handler(statement.type);
    if (handler) {
        // 直接调用execute_full方法执行整个指令
        handler->ExecPipe(this, statement);
    } else {
        std::cerr << "No handler found for statement type: "
                  << static_cast<int>(statement.type) << std::endl;
        state = EXIT;
    }

    // 提交 PC 变更（正常执行的唯一入口点）
    commit_pc();
}

// void ThreadContext::trace_instruction(StatementContext &statement) {
//     std::string opcode = S2s(statement.type);

//     // 使用DebugConfig获取完整的指令字符串（包含操作数）
//     std::string operands =
//         ptxsim::DebugConfig::get_full_instruction_string(statement);

//     // 使用PTX_TRACE_INSTR宏跟踪指令执行
//     PTX_TRACE_INSTR(pc, opcode, operands, BlockIdx, ThreadIdx);

//     // 记录性能统计
//     //
//     ptxsim::PTXDebugger::get().get_perf_stats().record_instruction(opcode);
// }

void ThreadContext::clear_temporaries() {
    while (!vecOp_phy_addrs.empty()) {
        vecOp_phy_addrs.pop();
    }
}

void ThreadContext::prepare_breakpoint_context(
    std::unordered_map<std::string, std::any> &context) {
    // 使用RegisterBankManager获取寄存器值
    if (register_bank_manager_) {
        // 计算warp_id和lane_id
        // int warp_id = (ThreadIdx.x + ThreadIdx.y * BlockDim.x +
        //                ThreadIdx.z * BlockDim.x * BlockDim.y) /
        //               WarpContext::WARP_SIZE;
        // int lane_id = (ThreadIdx.x + ThreadIdx.y * BlockDim.x +
        //                ThreadIdx.z * BlockDim.x * BlockDim.y) %
        //               WarpContext::WARP_SIZE;

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
    context["pc"] = get_pc();
    context["blockIdx"] = BlockIdx;
    context["threadIdx"] = ThreadIdx;
}

void ThreadContext::dump_state(std::ostream &os) const {
    os << "Thread State:" << std::endl;
    os << "  BlockIdx: [" << BlockIdx.x << ", " << BlockIdx.y << ", "
       << BlockIdx.z << "]" << std::endl;
    os << "  ThreadIdx: [" << ThreadIdx.x << ", " << ThreadIdx.y << ", "
       << ThreadIdx.z << "]" << std::endl;
    os << "  PC: " << get_pc() << std::endl;
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
    set_pc(0);
    set_next_pc(0);
    bar_id = 0;  // Reset barrier ID
    state = RUN;
    cc_reg = ConditionCodeRegister{}; // 重置条件码寄存器

    // 清空临时数据（寄存器管理由RegisterBankManager负责，无需本地重置）
    clear_temporaries();
    operand_collected.clear();
    operand_collected.resize(ThreadContext::MAX_OPERANDS_PER_INSTR);
    operand_is_immediate_.clear();
    operand_is_immediate_.resize(ThreadContext::MAX_OPERANDS_PER_INSTR);
}

// 添加新的执行方法
EXE_STATE ThreadContext::execute_thread_instruction() {
    this->_execute_once();
    return this->state; // 返回线程的实际状态
}

// acquire_operand() return the operand_phy_addr, which later use store in
// operand_collected by collect_operands()
void *ThreadContext::acquire_operand(const OperandContext &operand,
                                     const std::vector<Qualifier> &qualifiers) {
    switch (operand.kind()) {
    case OperandKind::VAR: {
        const VariableOperand &varOp = std::get<VariableOperand>(operand.data);
        // OperandContext::VAR *varOp = (OperandContext::VAR *)operand.data;

        // 优先在name2Share中查找（共享内存变量）
        // Always return the address of the offset value in the symbol table.
        // For 'mov %reg, symbol', this copies the offset into the register.
        // For memory operations, get_memory_addr() adds shared_mem_space.
        if (name2Share != nullptr) {
            auto share_it = name2Share->find(varOp.name);
            if (share_it != name2Share->end()) {
                PTX_DEBUG_EMU("Reading shared memory symbol: name=%s, "
                              "symbol_table_entry=%p, offset=0x%lx",
                              varOp.name.c_str(), share_it->second,
                              share_it->second->val);
                return &(share_it->second->val);
            }
        }

        // 尝试在CTAContext的name2Local中查找（本地内存变量）
        if (cta_context_ != nullptr) {
            auto local_it = cta_context_->name2Local.find(varOp.name);
            if (local_it != cta_context_->name2Local.end()) {
                // 对于本地内存变量，返回符号表条目的值（即实际内存地址）
                void *ret;

                if (local_mem_space != nullptr) {
                    ret = (void *)((uint64_t)local_mem_space +
                                   local_it->second->val);
                } else {
                    // 如果没有设置本地内存空间，则返回原始偏移量
                    ret = (void *)local_it->second->val;
                }
                PTX_DEBUG_EMU("Reading local memory from name2Local: name=%s, "
                              "symbol_table_entry=%p, stored_value=0x%lx, "
                              "local_mem_space=0x%lx",
                              varOp.name.c_str(), local_it->second, ret,
                              local_mem_space);
                return ret;
            }
        }

        // 如果在name2Share中没找到，再到name2Sym中查找（参数、局部变量等）
        auto sym_it = name2Sym->find(varOp.name);
        if (sym_it != name2Sym->end()) {
            // For PARAM space, return the GPU address directly so printf can read from GPU memory
            // For other spaces, return the address of the val field (for symbol addresses)
            if (varOp.name.find("param") != std::string::npos || 
                varOp.name.find("retval") != std::string::npos) {
            // For parameters/retval, return the GPU address where the value is stored
            // This way, when printf reads *formatPtrAddr, it reads from GPU memory
            return (void *)sym_it->second->val;
        }
        PTX_DEBUG_EMU("Reading kernel name2Sym from name2Sym: name=%s, "
                      "symbol_table_entry=%p, stored_value=0x%lx, "
                      "dereferenced_value=0x%lx",
                      varOp.name.c_str(), sym_it->second,
                      sym_it->second->val,
                      *(uint64_t *)(sym_it->second->val));
        return &(sym_it->second->val);
        }

        break;
    }

    case OperandKind::REG:
        return acquire_register(std::get<RegOperand>(operand.data), qualifiers);

    case OperandKind::ADDR:
        return get_memory_addr(std::get<AddrOperand>(operand.data), qualifiers);

    case OperandKind::IMM: {
        auto immOp = std::get<ImmOperand>(operand.data);
        Qualifier q = getDataQualifier(qualifiers);

        // 使用栈上缓冲区（每个 IMM 使用独立空间，支持多 IMM 指令）
        // 注意：此指针仅在当前指令执行期间有效！
        alignas(8) static thread_local char
            imm_buffer_pool[64][8]; // 支持最多 64 个 IMM/指令
        static thread_local int buffer_index = 0;

        // 使用模运算维护索引，避免溢出
        char *buffer = imm_buffer_pool[buffer_index];
        buffer_index = (buffer_index + 1) % 64;

        parseImmediate(immOp.value, q, buffer);
        return buffer;
    }

    case OperandKind::VEC: {
        auto vecOp = std::get<VecOperand>(operand.data);

        vecOp_phy_addrs.emplace();
        auto &stored_vec = vecOp_phy_addrs.back();

        for (auto &elem : vecOp.elements) {
            void *addr = acquire_operand(elem, qualifiers);
            if (!addr)
                return nullptr;
            elem.operand_phy_addr = addr;
            stored_vec.push_back(addr);
        }

        return stored_vec.data();
    }

    default:
        break;
    }

    return nullptr;
}

void ThreadContext::collect_operands(
    StatementContext &stmt, const std::vector<OperandContext> &operands,
    const std::vector<Qualifier> *qualifier) {
    // 获取每个操作数的字节大小
    std::vector<int> operand_bytes = getOperandBytes(*qualifier);

    // 扩展operand_collected向量以容纳所有操作数
    if (operand_collected.size() < operands.size()) {
        operand_collected.resize(operands.size());
        operand_is_immediate_.resize(operands.size());
    }

    for (int i = 0; i < operands.size(); i++) {
        // 获取当前操作数的字节大小，如果不存在则使用最后一个元素
        int bytes = operand_bytes.size() > i
                        ? operand_bytes[i]
                        : (operand_bytes.empty() ? 0 : operand_bytes.back());

        trace_status(ptxsim::log_level::debug, "thread", "Collect: %s ",
                     operands[i].toString(bytes).c_str());

        // Track whether this operand is immediate
        // For immediate operands: operand_collected[i] is a pointer to the immediate value
        // For register/variable operands: operand_collected[i] is the actual value/address
        operand_is_immediate_[i] = (operands[i].kind() == OperandKind::IMM);

        // 获取当前操作数的物理地址
        operand_collected[i] = operands[i].operand_phy_addr;
    }
    // FIXME should use stmt qualifier?
    // stmt.qualifier = *qualifier;
};

void ThreadContext::commit_operand(StatementContext &stmt,
                                   const OperandContext &operand,
                                   const std::vector<Qualifier> &qualifier) {
    int bytes = getBytes(qualifier);
    trace_status(ptxsim::log_level::debug, "thread", "Commit:  %s ",
                 operand.toString(bytes).c_str());
};

void *ThreadContext::acquire_register(const RegOperand &reg,
                                      std::vector<Qualifier> qualifier) {
    // 检查是否是特殊寄存器
    if (reg.name.find('.') != std::string::npos) {
        if (reg.name == "tid.x")
            return &ThreadIdx.x;
        if (reg.name == "tid.y")
            return &ThreadIdx.y;
        if (reg.name == "tid.z")
            return &ThreadIdx.z;
        if (reg.name == "ctaid.x")
            return &BlockIdx.x;
        if (reg.name == "ctaid.y")
            return &BlockIdx.y;
        if (reg.name == "ctaid.z")
            return &BlockIdx.z;
        if (reg.name == "nctaid.x")
            return &GridDim.x;
        if (reg.name == "nctaid.y")
            return &GridDim.y;
        if (reg.name == "nctaid.z")
            return &GridDim.z;
        if (reg.name == "ntid.x")
            return &BlockDim.x;
        if (reg.name == "ntid.y")
            return &BlockDim.y;
        if (reg.name == "ntid.z")
            return &BlockDim.z;
    }

    // 确保register_bank_manager_存在
    if (!register_bank_manager_) {
        throw std::runtime_error("RegisterBankManager is required but not set");
    }

    std::string combinedName = reg.fullName();
    void *reg_data =
        register_bank_manager_->get_register(combinedName, warp_id_, lane_id_);

    if (reg_data == nullptr) {
      throw InvalidMemoryAccessException(0, 0, "null register data",
          "Register not found in bank manager: " + combinedName);
    }

    return reg_data;
}

void *ThreadContext::get_memory_addr(const AddrOperand &fa,
                                     const std::vector<Qualifier> &qualifiers) {
    void *ret;
    if (fa.offsetType == AddrOperand::OffsetType::REGISTER) {
        // 1. 执行到get_memory_addr时，传入的qualifiers含有Q_GLOBAL, Q_SHARED,
        // Q_PARAM如何处理地址信息，
        // 把qualifiers里的地址信息设定到mem_qualifiers，而不是假定哟个Q_U64.
        Qualifier mem_qualifier = Qualifier::Q_UNKNOWN;
        for (const auto &q : qualifiers) {
            // 将地址空间信息添加到mem_qualifiers中
            if (q == Qualifier::Q_SHARED) {
                mem_qualifier = Qualifier::Q_U32;
            } else if (q == Qualifier::Q_GLOBAL || q == Qualifier::Q_PARAM ||
                       q == Qualifier::Q_LOCAL) {
                mem_qualifier = Qualifier::Q_U64;
            }
        }
        if (mem_qualifier == Qualifier::Q_UNKNOWN) {
            throw ExecutionStateException(
                0, "unknown memory qualifier",
                "Failed to determine memory address space from qualifiers");
        }

        assert(fa.registerOffset->kind() == OperandKind::REG);

        void *regAddr = acquire_register(
            std::get<RegOperand>(fa.registerOffset->data), {mem_qualifier});
        if (!regAddr)
            return nullptr;

        uint64_t reg_value;
        if (mem_qualifier == Qualifier::Q_U32) {
            reg_value = *(uint32_t *)regAddr;
        } else {
            reg_value = *(uint64_t *)regAddr;
        }

        // 如果是shared memory访问，需要特殊处理
        if (QvecHasQ(qualifiers, Qualifier::Q_SHARED)) {
            // 对于共享内存访问，寄存器中的值是偏移量，需要加上共享内存基地址
            if (shared_mem_space != nullptr) {
                uint64_t offset = reg_value;
                // 查name2Share获取baseSymbol在共享内存中的偏移量
                if (name2Share != nullptr && !fa.baseSymbol.empty()) {
                    auto it = name2Share->find(fa.baseSymbol);
                    if (it != name2Share->end()) {
                        offset += it->second->val;
                    }
                }
                ret = (void *)((uint64_t)shared_mem_space + offset);
            } else {
                // 如果没有设置共享内存基地址，则返回nullptr
                return nullptr;
            }
            // } else if (QvecHasQ(qualifiers, Qualifier::Q_LOCAL)) {
            //     //
            //     对于本地内存访问，寄存器中的值是偏移量，需要加上本地内存基地址
            //     if (local_mem_space != nullptr) {
            //         ret = (void *)((uint64_t)local_mem_space + reg_value);
            //     } else {
            //         // 如果没有设置本地内存基地址，则返回nullptr
            //         return nullptr;
            //     }
        } else {
            ret = (void *)reg_value;
        }
    } else {
        // 直接通过ID查找符号表或共享内存；如果ID为空，回退到baseSymbol
        const std::string &lookupName =
            fa.id.empty() ? fa.baseSymbol : fa.id;

        // get_memory_addr debug logs - disabled for clarity
#if 0
        PTX_DEBUG_MEM("get_memory_addr lookup: id=%s base=%s lookup=%s qualifiers=%zu",
                      fa.id.c_str(), fa.baseSymbol.c_str(),
                      lookupName.c_str(), qualifiers.size());
#endif

        // [REFACT] Check if lookupName is a register name (for handling [%rd4+4] register base + immediate offset)
        // NOTE: This assumes register names do not conflict with symbol names in name2Sym/name2Share
        RegOperand regOp;
        if (ptx::syntax::parseRegisterFromText(lookupName, regOp)) {
            // lookupName is a register, read base address from register bank
#if 0
            PTX_DEBUG_MEM("Address base is a register: %s, fetching from register bank", lookupName.c_str());
#endif

            // Determine qualifier for register data type
            Qualifier mem_qualifier = Qualifier::Q_UNKNOWN;
            for (const auto &q : qualifiers) {
                if (q == Qualifier::Q_SHARED) {
                    mem_qualifier = Qualifier::Q_U32;
                } else if (q == Qualifier::Q_GLOBAL || q == Qualifier::Q_PARAM ||
                           q == Qualifier::Q_LOCAL) {
                    mem_qualifier = Qualifier::Q_U64;
                }
            }
            if (mem_qualifier == Qualifier::Q_UNKNOWN) {
                mem_qualifier = Qualifier::Q_U64; // default to 64-bit
            }

            void *regAddr = acquire_register(regOp, {mem_qualifier});
            if (!regAddr) {
                PTX_DEBUG_EMU("Failed to acquire register: %s", lookupName.c_str());
                return nullptr;
            }

            // For shared memory address calculation, we need to handle signed offsets correctly.
            // PTX allows negative offsets in address expressions like [%r5+124] where %r5 can be negative.
            // Read register value as signed 32-bit for shared memory, then sign-extend to 64-bit.
            int64_t base_value;
            if (QvecHasQ(qualifiers, Qualifier::Q_SHARED) && mem_qualifier == Qualifier::Q_U32) {
                // For shared memory with 32-bit registers, read as signed to support negative offsets
                base_value = (int64_t)*(int32_t *)regAddr;
            } else {
                base_value = (mem_qualifier == Qualifier::Q_U32)
                    ? (int64_t)*(uint32_t *)regAddr
                    : (int64_t)*(uint64_t *)regAddr;
            }

            // For shared memory, add shared_mem_space to the base value
            if (QvecHasQ(qualifiers, Qualifier::Q_SHARED)) {
                if (shared_mem_space != nullptr) {
                    ret = (void *)((uint64_t)shared_mem_space + base_value);
                } else {
                    PTX_DEBUG_EMU("get_memory_addr: Q_SHARED but shared_mem_space is null!");
                    return nullptr;
                }
            } else {
                ret = (void *)base_value;
            }
            PTX_DEBUG_EMU("Register %s contains base value: 0x%lx, final ret: %p", lookupName.c_str(), base_value, ret);

            // Skip symbol table lookup, jump directly to offset handling
            goto handle_offset;
        }

        auto sym_it = name2Sym->find(lookupName);
        if (sym_it != name2Sym->end()) {
#if 0
            PTX_DEBUG_MEM("Reading kernel argument from name2Sym in "
                          "get_memory_addr: name=%s, "
                          "symbol_table_entry=%p, stored_value=0x%lx",
                          lookupName.c_str(), sym_it->second,
                          sym_it->second->val);
#endif
            ret = (void *)sym_it->second->val;
        } else if (name2Share != nullptr) {
            // 如果在name2Sym中没找到，继续在name2Share中查找
            auto share_it = name2Share->find(lookupName);
            if (share_it != name2Share->end()) {
#if 0
                PTX_DEBUG_MEM("Reading shared memory from name2Share in "
                              "get_memory_addr: name=%s, "
                              "symbol_table_entry=%p, stored_value=0x%lx",
                              lookupName.c_str(), share_it->second,
                              share_it->second->val);
#endif

                // 修正：对于共享内存变量，应该返回相对于共享内存空间的绝对地址
                if (shared_mem_space != nullptr) {
                    ret = (void *)((uint64_t)shared_mem_space +
                                   share_it->second->val);
                } else {
                    // 如果没有设置共享内存空间，则返回原始偏移量
                    ret = (void *)share_it->second->val;
                }
            } else {
                // 检查是否是本地内存变量
                auto local_it = cta_context_->name2Local.find(lookupName);
                if (local_it != cta_context_->name2Local.end()) {
                    // 对于本地内存变量，应该返回相对于当前线程本地内存空间的绝对地址
                    // 直接使用当前线程的本地内存空间（已经通过set_local_memory_space设置）
                    if (local_mem_space != nullptr) {
                        ret = (void *)((uint64_t)local_mem_space +
                                       local_it->second->val);
                    } else {
                        // 如果没有设置本地内存空间，则返回原始偏移量
                        ret = (void *)local_it->second->val;
                    }
                    PTX_DEBUG_EMU("Reading local memory from name2Local in "
                                  "get_memory_addr: name=%s, "
                                  "symbol_table_entry=%p, stored_value=0x%lx, "
                                  "local_mem_space=0x%lx",
                                  lookupName.c_str(), local_it->second, ret,
                                  local_mem_space);

                } else {
                    // 对于本地内存访问，如果在name2Local中没找到，说明可能尚未初始化
                    PTX_DEBUG_EMU(
                        "Local memory variable not found in name2Local: %s",
                        lookupName.c_str());
                    return nullptr;
                }
            }
        } else {
            // 如果都没找到，返回nullptr
            PTX_DEBUG_EMU("get_memory_addr symbol lookup failed: lookup=%s",
                          lookupName.c_str());
            return nullptr;
        }
    }

    // 如果是shared memory访问，需要特殊处理
    if (QvecHasQ(qualifiers, Qualifier::Q_SHARED)) {
        // 对于共享内存，地址已经在上面的逻辑中正确处理了
    } else if (QvecHasQ(qualifiers, Qualifier::Q_LOCAL)) {
        // 对于本地内存，地址也已经在上面的逻辑中正确处理了
    }

    handle_offset:
    // 处理偏移量
    if (!fa.immediateOffset.empty()) {
        // 直接解析偏移量字符串，避免创建临时立即数操作数
        int64_t offset = 0;
        try {
            // 解析偏移量字符串为整数值
            offset = std::stoll(fa.immediateOffset);
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
      if (address >> 32 != SHMEMADDR) {
        throw InvalidMemoryAccessException(
            address, 0, "invalid shared memory address",
            "Address high bits do not match SHMEMADDR constant");
      }
    } else {
      SHMEMADDR = address >> 32;  // 只保存高32位
    }
}

void ThreadContext::mov(void *from, void *to, const std::vector<Qualifier> &q) {
    int bytes = getBytes(q);
    memcpy(to, from, bytes);
}

bool ThreadContext::isIMMorVEC(OperandContext &op) {
    return (op.kind() == OperandKind::IMM || op.kind() == OperandKind::VEC);
}

bool ThreadContext::is_immediate_or_vector(OperandContext &op) {
    return (op.kind() == OperandKind::IMM || op.kind() == OperandKind::VEC);
}

void ThreadContext::print_instruction_status(StatementContext &stmt) {
    // 获取操作数字符串
    std::string operands_str = ptxsim::DebugConfig::getOperandsString(stmt);

    // 获取操作码字符串
    std::string opcode_str = S2s(stmt.type);

    // 使用trace_status函数替代PTX_TRACE宏
    trace_status(ptxsim::log_level::trace, "instr", "PC[0x%x] %s %s", get_pc(),
                 opcode_str.c_str(), operands_str.c_str());
}

// 【Stage 4】从 warp_state 同步状态到 ThreadContext
void ThreadContext::sync_from_warp_state() {
    if (!warp_context_) return;

    int lane_id = lane_id_;
    if (lane_id < 0 || lane_id >= WarpContext::WARP_SIZE) return;

    ptxsim::ThreadState& thread_state = warp_context_->get_warp_state().threads[lane_id];

    // PC 通过 get_pc()/get_next_pc() 直接从 warp_state 读取，无需同步
    // sync_to_warp_state() 的 next_pc 同步保持一致性

    // 同步状态
    switch (thread_state.status) {
        case ptxsim::ThreadStatus::Active:
            state = RUN;
            break;
        case ptxsim::ThreadStatus::Blocked:
            state = BAR_SYNC;
            break;
        case ptxsim::ThreadStatus::Exited:
            state = EXIT;
            break;
        case ptxsim::ThreadStatus::Yielded:
            // Yielded 状态暂时映射到 RUN
            state = RUN;
            break;
    }
}

// 【Stage 4】将 ThreadContext 的状态同步到 warp_state
void ThreadContext::sync_to_warp_state() {
    if (!warp_context_) return;

    int lane_id = lane_id_;
    if (lane_id < 0 || lane_id >= WarpContext::WARP_SIZE) return;


    ptxsim::ThreadState& thread_state = warp_context_->get_warp_state().threads[lane_id];

    // 如果线程已经在 barrier 等待（is_blocked=true 或 status=Blocked），
    // 则只同步 next_pc，不覆盖 blocked 状态。
    // 注意：如果 barrier 主动通过 set_state(RUN) + 清除 is_blocked 来释放线程，
    // 则应该在调用本函数之前清除 is_blocked（参见 sm_context.cpp synchronize_barrier）。
    bool already_blocked = (thread_state.is_blocked ||
                            thread_state.status == ptxsim::ThreadStatus::Blocked);

    // 屏障完成处理会通过 warp_ctx->advance_thread_pc() 或 force_set_pc() 直接更新 warp_state
    // 此处只同步 ThreadContext 自己维护的 next_pc 状态
    thread_state.next_pc = get_next_pc();

    // 如果已经 blocked，只同步 next_pc，不修改状态
    if (already_blocked) {
        return;
    }

    // 同步状态
    switch (state) {
        case RUN:
            thread_state.status = ptxsim::ThreadStatus::Active;
            thread_state.is_blocked = false;
            thread_state.is_active = true;
            break;
        case BAR_SYNC:
            thread_state.status = ptxsim::ThreadStatus::Blocked;
            thread_state.is_blocked = true;
            break;
        case EXIT:
            thread_state.status = ptxsim::ThreadStatus::Exited;
            thread_state.is_exited = true;
            thread_state.is_active = false;
            thread_state.is_blocked = false;
            break;
        default:
            break;
    }
}

// PC accessors - delegate to WarpState via WarpContext
int ThreadContext::get_pc() const {
    if (!warp_context_) return 0;
    int lane = lane_id_;
    if (lane < 0 || lane >= 32) return 0;
    return warp_context_->get_warp_state().threads[lane].pc;
}

void ThreadContext::set_pc(int new_pc) {
    if (!warp_context_) return;
    int lane = lane_id_;
    if (lane < 0 || lane >= 32) return;
    warp_context_->get_warp_state().threads[lane].pc = new_pc;
    warp_context_->get_warp_state().threads[lane].next_pc = new_pc;
}

int ThreadContext::get_next_pc() const {
    if (!warp_context_) return 0;
    int lane = lane_id_;
    if (lane < 0 || lane >= 32) return 0;
    return warp_context_->get_warp_state().threads[lane].next_pc;
}

void ThreadContext::set_next_pc(int new_next_pc) {
    if (!warp_context_) return;
    int lane = lane_id_;
    if (lane < 0 || lane >= 32) return;
    warp_context_->get_warp_state().threads[lane].next_pc = new_next_pc;
}

void ThreadContext::force_set_pc(int new_pc) {
    if (!warp_context_) return;
    int lane = lane_id_;
    if (lane < 0 || lane >= 32) return;
    warp_context_->get_warp_state().threads[lane].pc = new_pc;
}

void ThreadContext::commit_pc() {
    set_pc(get_next_pc());
}
