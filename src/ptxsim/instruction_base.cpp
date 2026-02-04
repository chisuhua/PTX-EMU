#include <cstdio>
#include "ptxsim/instruction_base.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/thread_context.h"

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
    context->next_pc = context->pc + 1;
}

// Simple handlers (labels, pragmas, dollar names)
void SimpleHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {

    context->next_pc = context->pc + 1;
}

// Void instructions (ret, exit, trap, etc.)
void VoidHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    context->trace_status(ptxsim::log_level::debug, "thread", 
                          "PC=%x VOID_INSTR: %s", context->pc, 
                          stmt.instructionText.c_str());
    context->next_pc = context->pc + 1;
}

// Branch instructions
void BranchHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    context->trace_status(ptxsim::log_level::debug, "thread", 
                          "PC=%x BRANCH: %s", context->pc, 
                          stmt.instructionText.c_str());
    const BranchInstr &branchInstr = std::get<BranchInstr>(stmt.data);
    executeBranch(context, branchInstr);
}

// Barrier instructions
void BarrierHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    context->trace_status(ptxsim::log_level::debug, "thread", 
                          "PC=%x BARRIER: %s", context->pc, 
                          stmt.instructionText.c_str());
    const BarrierInstr &barrierInstr = std::get<BarrierInstr>(stmt.data);
    executeBarrier(context, barrierInstr);
}

// Call instructions
void CallHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    context->trace_status(ptxsim::log_level::debug, "thread", 
                          "PC=%x CALL: %s", context->pc, 
                          stmt.instructionText.c_str());
    const CallInstr &callInstr = std::get<CallInstr>(stmt.data);
    executeCall(context, callInstr);
}

// Pipeline Handler Implementation
void PipelineHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    switch (stmt.state) {
        case InstructionState::READY:
            if (!prepareOperands(context, stmt)) {
                return;
            }
            stmt.state = InstructionState::PREPARE;
            break;
            
        case InstructionState::PREPARE:
            if (!executeOperation(context, stmt)) {
                return;
            }
            stmt.state = InstructionState::EXECUTE;
            break;
            
        case InstructionState::EXECUTE:
            if (!commitResults(context, stmt)) {
                return;
            }
            stmt.state = InstructionState::COMMIT;
            context->next_pc = context->pc + 1;
            break;
            
        case InstructionState::COMMIT:
            // Should not reach here in normal flow
            stmt.state = InstructionState::READY;
            break;
    }
}

bool PipelineHandler::acquireAllOperands(ThreadContext *context, 
                                       std::vector<OperandContext> &operands, 
                                       const std::vector<Qualifier> &qualifiers, 
                                       int opCount) {
    for (int i = 0; i < opCount && i < static_cast<int>(operands.size()); i++) {
        if (!operands[i].operand_phy_addr) {
            void *result = context->acquire_operand(operands[i], qualifiers);
            if (!result) {
                PTX_DEBUG_EMU("Failed to get operand address for op[%d]", i);
                return false;
            }
            operands[i].setPhyAddr(result);
        }
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
    processOperation(context, &(context->operand_collected[0]), instr.qualifiers);
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
                          "PC=%x CP_ASYNC: %s", context->pc,
                          stmt.instructionText.c_str());
    const CpAsyncInstr &cpAsyncInstr = std::get<CpAsyncInstr>(stmt.data);
    executeAsyncCopy(context, cpAsyncInstr);
    context->next_pc = context->pc + 1;
}
