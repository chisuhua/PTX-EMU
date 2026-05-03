#include <cstdio>
#include "ptxsim/instruction_base.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"

// Define PTX_DEBUG_EMU if not already defined
#ifndef PTX_DEBUG_EMU
#include <cstdarg>
inline void ptx_debug_emu_impl(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[PTX_DEBUG] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}
#define PTX_DEBUG_EMU(...) ptx_debug_emu_impl(__VA_ARGS__)
#endif

// Declaration handlers (variable declarations, etc.)
void DeclarationHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    // Declarations are handled at kernel initialization, not during execution
    context->set_next_pc(context->get_pc() + 1);
}

// Simple handlers (labels, pragmas, dollar names)
void SimpleHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {

    context->set_next_pc(context->get_pc() + 1);
}

// Void instructions (ret, exit, trap, etc.)
void VoidHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    context->trace_status(ptxsim::log_level::debug, "thread", 
                          "PC=%x VOID_INSTR: %s", context->get_pc(), 
                          stmt.instructionText.c_str());
    processOperation(context, stmt);
    context->set_next_pc(context->get_pc() + 1);
}

// Branch instructions
void BranchHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    context->trace_status(ptxsim::log_level::debug, "thread", 
                          "PC=%x BRANCH: %s", context->get_pc(), 
                          stmt.instructionText.c_str());
    const BranchInstr &branchInstr = std::get<BranchInstr>(stmt.data);
    executeBranch(context, branchInstr);
}

// Barrier instructions
void BarrierHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    context->trace_status(ptxsim::log_level::debug, "thread", 
                          "PC=%x BARRIER: %s", context->get_pc(), 
                          stmt.instructionText.c_str());
    const BarrierInstr &barrierInstr = std::get<BarrierInstr>(stmt.data);
    executeBarrier(context, barrierInstr);
}

// Call instructions
void CallBaseHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    context->trace_status(ptxsim::log_level::debug, "thread", 
                          "PC=%x CALL: %s", context->get_pc(), 
                          stmt.instructionText.c_str());
    const CallInstr &callInstr = std::get<CallInstr>(stmt.data);
    executeCall(context, callInstr);
    // Default behavior: advance to next instruction
    // Derived classes may override this to set next_pc to target address
    context->set_next_pc(context->get_pc() + 1);
}

// Pipeline Handler Implementation
void PipelineHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    // 保存当前PC，避免屏障处理器修改后 get_pc() 返回错误值
    int saved_pc = context->get_pc();

    // 流水线阶段1：获取操作数 - 每个线程独立运行，返回false表示资源冲突需要重试
    // 确保准备操作数的过程独立运行，避免线程间状态干扰
    if (!prepareOperands(context, stmt)) {
        PTX_DEBUG_EMU("[PTX_PIPELINE_RETRY] stage=prepare pc=%d instr=%s",
                      context->get_pc(), stmt.instructionText.c_str());
        return;
    }

    if (!executeOperation(context, stmt)) {
        PTX_DEBUG_EMU("[PTX_PIPELINE_RETRY] stage=execute pc=%d instr=%s",
                      context->get_pc(), stmt.instructionText.c_str());
        return;
    }

    if (!commitResults(context, stmt)) {
        PTX_DEBUG_EMU("[PTX_PIPELINE_RETRY] stage=commit pc=%d instr=%s",
                      context->get_pc(), stmt.instructionText.c_str());
        return;
    }

    // stmt.state is shared across all threads in a CTA; do not write to it
    // here to avoid a data race. The state begins as READY and need not be
    // reset—each thread drives its own pipeline atomically per ExecPipe call.
    // 使用保存的PC而不是 get_pc()，因为屏障处理器可能已经通过 set_thread_pc 修改了PC
    context->set_next_pc(saved_pc + 1);
}

bool PipelineHandler::acquireAllOperands(ThreadContext *context, 
                                       std::vector<OperandContext> &operands, 
                                       const std::vector<Qualifier> &qualifiers, 
                                       int opCount) {
    for (int i = 0; i < opCount && i < static_cast<int>(operands.size()); i++) {
        void *result = context->acquire_operand(operands[i], qualifiers);
        if (!result) {
            PTX_DEBUG_EMU("Failed to get operand address for op[%d]", i);
            PTX_DEBUG_EMU("  pc=%d op=%s", context->get_pc(),
                          operands[i].toString().c_str());
            if (operands[i].kind() == OperandKind::ADDR) {
                const auto &addr = std::get<AddrOperand>(operands[i].data);
                const char *offsetType =
                    addr.offsetType == AddrOperand::OffsetType::REGISTER
                        ? "REGISTER"
                        : "IMMEDIATE";
                std::string regText = "<null>";
                if (addr.registerOffset) {
                    regText = addr.registerOffset->toString();
                }
                PTX_DEBUG_EMU(
                    "  addr_fields: id=%s base=%s offsetType=%s imm=%s regOffset=%s",
                    addr.id.c_str(), addr.baseSymbol.c_str(), offsetType,
                    addr.immediateOffset.c_str(), regText.c_str());
            }
            return false;
        }
        operands[i].setPhyAddr(result);
    }
    return true;
}

void PipelineHandler::releaseAllOperands(std::vector<OperandContext> &operands, int opCount) {
    for (int i = 0; i < opCount && i < static_cast<int>(operands.size()); i++) {
        operands[i].setPhyAddr(nullptr);
    }
}

// Generic Pipeline Handler
bool GenericPipelineHandler::prepareOperands(ThreadContext *context, StatementContext &stmt) {
    GenericInstr &instr = std::get<GenericInstr>(stmt.data);
    if (!acquireAllOperands(context, instr.operands, instr.qualifiers, 
                           static_cast<int>(instr.operands.size()))) {
        return false;
    }
    context->collect_operands(stmt, instr.operands, &(instr.qualifiers));
    return true;
}

bool GenericPipelineHandler::executeOperation(ThreadContext *context, StatementContext &stmt) {
    const GenericInstr &instr = std::get<GenericInstr>(stmt.data);

    if (stmt.type == S_SETP && instr.operands.size() >= 3) {
        void *dst = instr.operands[0].operand_phy_addr;
        void *src1 = instr.operands[1].operand_phy_addr;
        void *src2 = instr.operands[2].operand_phy_addr;
        // No-op: dst is used directly in the write below
        Qualifier cmpOp = getCmpOpQualifier(instr.qualifiers);
        Qualifier dtype = getDataQualifier(instr.qualifiers);

        auto cmp = [&](auto a, auto b) {
            switch (cmpOp) {
            case Qualifier::Q_EQ: return a == b;
            case Qualifier::Q_NE: return a != b;
            case Qualifier::Q_LT: return a < b;
            case Qualifier::Q_LE: return a <= b;
            case Qualifier::Q_GT: return a > b;
            case Qualifier::Q_GE: return a >= b;
            default: return false;
            }
        };

        bool cmp_result = false;
        if (src1 == nullptr || src2 == nullptr) {
            cmp_result = false;
        } else switch (dtype) {
        case Qualifier::Q_U8: cmp_result = cmp(*static_cast<uint8_t*>(src1), *static_cast<uint8_t*>(src2)); break;
        case Qualifier::Q_S8: cmp_result = cmp(*static_cast<int8_t*>(src1), *static_cast<int8_t*>(src2)); break;
        case Qualifier::Q_U16: cmp_result = cmp(*static_cast<uint16_t*>(src1), *static_cast<uint16_t*>(src2)); break;
        case Qualifier::Q_S16: cmp_result = cmp(*static_cast<int16_t*>(src1), *static_cast<int16_t*>(src2)); break;
        case Qualifier::Q_U32: cmp_result = cmp(*static_cast<uint32_t*>(src1), *static_cast<uint32_t*>(src2)); break;
        case Qualifier::Q_S32: cmp_result = cmp(*static_cast<int32_t*>(src1), *static_cast<int32_t*>(src2)); break;
        case Qualifier::Q_F32: cmp_result = cmp(*static_cast<float*>(src1), *static_cast<float*>(src2)); break;
        default: break;
        }

        if (dst != nullptr) {
            *static_cast<uint8_t *>(dst) = cmp_result ? 1 : 0;
        }
        return true;
    }

    processOperation(context, &(context->operand_collected[0]), instr.qualifiers,
                     &context->operand_is_immediate_);
    return true;
}

bool GenericPipelineHandler::commitResults(ThreadContext *context, StatementContext &stmt) {
    GenericInstr &instr = std::get<GenericInstr>(stmt.data);
    // Note: PTX generic instructions have exactly one destination operand at index 0.
    if (!instr.operands.empty()) {
        context->commit_operand(stmt, instr.operands[0], instr.qualifiers);
    }
    releaseAllOperands(instr.operands, static_cast<int>(instr.operands.size()));
    return true;
}

// Atomic Pipeline Handler
bool AtomicPipelineHandler::prepareOperands(ThreadContext *context, StatementContext &stmt) {
    AtomInstr &instr = std::get<AtomInstr>(stmt.data);
    if (!acquireAllOperands(context, instr.operands, instr.qualifiers, 
                           static_cast<int>(instr.operands.size()))) {
        return false;
    }
    context->collect_operands(stmt, instr.operands, &(instr.qualifiers));
    return true;
}

bool AtomicPipelineHandler::executeOperation(ThreadContext *context, StatementContext &stmt) {
    const AtomInstr &instr = std::get<AtomInstr>(stmt.data);
    processAtomicOperation(context, &(context->operand_collected[0]), instr.qualifiers);
    return true;
}

bool AtomicPipelineHandler::commitResults(ThreadContext *context, StatementContext &stmt) {
    AtomInstr &instr = std::get<AtomInstr>(stmt.data);
    if (!instr.operands.empty()) {
        context->commit_operand(stmt, instr.operands[0], instr.qualifiers);
    }
    releaseAllOperands(instr.operands, static_cast<int>(instr.operands.size()));
    return true;
}

// WMMA Pipeline Handler
bool WmmaPipelineHandler::prepareOperands(ThreadContext *context, StatementContext &stmt) {
    WmmaInstr &instr = std::get<WmmaInstr>(stmt.data);
    if (!acquireAllOperands(context, instr.operands, instr.qualifiers, 
                           static_cast<int>(instr.operands.size()))) {
        return false;
    }
    context->collect_operands(stmt, instr.operands, &(instr.qualifiers));
    return true;
}

bool WmmaPipelineHandler::executeOperation(ThreadContext *context, StatementContext &stmt) {
    const WmmaInstr &instr = std::get<WmmaInstr>(stmt.data);
    processWmmaOperation(context, &(context->operand_collected[0]), instr.qualifiers);
    return true;
}

bool WmmaPipelineHandler::commitResults(ThreadContext *context, StatementContext &stmt) {
    WmmaInstr &instr = std::get<WmmaInstr>(stmt.data);
    if (!instr.operands.empty()) {
        context->commit_operand(stmt, instr.operands[0], instr.qualifiers);
    }
    releaseAllOperands(instr.operands, static_cast<int>(instr.operands.size()));
    return true;
}

void AsyncCopyHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    context->trace_status(ptxsim::log_level::debug, "thread",
                          "PC=%x CP_ASYNC: %s", context->get_pc(),
                          stmt.instructionText.c_str());
    const CpAsyncInstr &cpAsyncInstr = std::get<CpAsyncInstr>(stmt.data);
    executeAsyncCopy(context, cpAsyncInstr);
    context->set_next_pc(context->get_pc() + 1);
}
