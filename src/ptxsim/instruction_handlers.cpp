#include "ptxsim/instruction_handlers.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include "utils/logger.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// 前向声明MemoryManager
class MemoryManager;

#define IMPLEMENT_ABI_DIRECTIVE(Name)                                          \
    void Name::ExecPipe(ThreadContext *context, StatementContext &stmt) {      \
        context->next_pc = context->pc + 1;                                    \
        return;                                                                \
    }

#define IMPLEMENT_OPERAND_REG(Name)                                            \
    void Name::ExecPipe(ThreadContext *context, StatementContext &stmt) {      \
        context->next_pc = context->pc + 1;                                    \
        return;                                                                \
    }

#define IMPLEMENT_OPERAND_CONST(Name)                                          \
    void Name::ExecPipe(ThreadContext *context, StatementContext &stmt) {      \
        assert(false);                                                         \
        context->next_pc = context->pc + 1;                                    \
        return; /* Return true to satisfy bool return type */                  \
    }

#define IMPLEMENT_OPERAND_MEMORY(Name)                                         \
    void Name::ExecPipe(ThreadContext *context, StatementContext &stmt) {      \
        context->next_pc = context->pc + 1;                                    \
        return;                                                                \
    }

// dollar TODO
#define IMPLEMENT_SIMPLE_NAME(Name)                                            \
    void Name::ExecPipe(ThreadContext *context, StatementContext &stmt) {      \
        context->next_pc = context->pc + 1;                                    \
        return;                                                                \
    }

// PRAGMA TODO
#define IMPLEMENT_SIMPLE_STRING(Name)                                          \
    void Name::ExecPipe(ThreadContext *context, StatementContext &stmt) {      \
        context->next_pc = context->pc + 1;                                    \
        return;                                                                \
    }

// VOID_INSTR
#define IMPLEMENT_VOID_INSTR(Name)                                             \
    void Name::ExecPipe(ThreadContext *context, StatementContext &stmt) {      \
        context->trace_status(ptxsim::log_level::debug, "thread",              \
                              "PC=%x " #Name " : %s", context->pc,             \
                              stmt.instructionText.c_str());                   \
        process_operation(context);                                            \
        return;                                                                \
    }

// BRANCH
#define IMPLEMENT_BRANCH(Name)                                                 \
    bool Name::operate(ThreadContext *context, StatementContext &stmt) {       \
        context->trace_status(ptxsim::log_level::debug, "thread",              \
                              "PC=%x " #Name " : %s", context->pc,             \
                              stmt.instructionText.c_str());                   \
        const BranchInstr &ss = std::get<BranchInstr>(stmt.data);              \
        auto iter = context->label2pc.find(ss.target);                         \
        assert(iter != context->label2pc.end());                               \
        void *op[0];                                                           \
        op[0] = &(iter->second);                                               \
        process_operation(context, op, ss.qualifiers);                         \
        return true;                                                           \
    }

// BARRIER
#define IMPLEMENT_BARRIER(Name)                                                \
    bool Name::operate(ThreadContext *context, StatementContext &stmt) {       \
        context->trace_status(ptxsim::log_level::debug, "thread",              \
                              "PC=%x " #Name " : %s", context->pc,             \
                              stmt.instructionText.c_str());                   \
        const BarrierInstr &ss = std::get<BarrierInstr>(stmt.data);            \
                                                                               \
        if (ss.type == "cta") {                                                \
            context->state = BAR_SYNC;                                         \
            process_operation(context, *ss.barId, ss.qualifiers);              \
        } else {                                                               \
            context->state = BAR_SYNC;                                         \
            process_operation(context, *ss.barId, ss.qualifiers);              \
            PTX_DEBUG_EMU("Unsupported barrier type %s", ss.type.c_str());     \
        }                                                                      \
        return true;                                                           \
    }

// GENERIC_INSTR
#define IMPLEMENT_GENERIC_INSTR(Name)                                          \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
        context->trace_status(ptxsim::log_level::debug, "thread",              \
                              "PC=%x " #Name " : %s", context->pc,             \
                              stmt.instructionText.c_str());                   \
        const GenericInstr &ss = std::get<GenericInstr>(stmt.data);            \
        /* Pre-validate operand addresses */                                   \
        for (int i = 0; i < op_count; i++) {                                   \
            if (!ss.operands[i].operand_phy_addr) {                            \
                void *result =                                                 \
                    context->acquire_operand(ss.operands[i], ss.qualifiers);   \
                if (!result) {                                                 \
                    PTX_DEBUG_EMU("Failed to get operand address for op[%d]",  \
                                  i);                                          \
                    return false;                                              \
                } else {                                                       \
                    ss.operands[i].operand_phy_addr = result;                  \
                }                                                              \
            }                                                                  \
        }                                                                      \
        context->collect_operands(stmt, ss.operands, &(ss.qualifiers));        \
        context->next_pc = context->pc + 1;                                    \
        return true;                                                           \
    }                                                                          \
    bool Name::commit(ThreadContext *context, StatementContext &stmt) {        \
        const GenericInstr &ss = std::get<GenericInstr>(stmt.data);            \
        context->commit_operand(stmt, ss.operands[0], ss.qualifiers);          \
        for (int i = 0; i < op_count; i++) {                                   \
            ss.operands[i].operand_phy_addr = nullptr;                         \
        }                                                                      \
        stmt.state = InstructionState::COMMIT;                                 \
        return true; /* Typically no commit work needed */                     \
    }

// AT BRA
#define IMPLEMENT_PREDICATE_PREFIX(Name)                                       \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
        context->trace_status(ptxsim::log_level::debug, "thread",              \
                              "PC=%x " #Name " : %s", context->pc,             \
                              stmt.instructionText.c_str());                   \
        const PredicatePrefix &ss = std::get<PredicatePrefix>(stmt.data);      \
        /* Pre-validate predicate operand addresses */                         \
        if (!ss.operands[0].operand_phy_addr) {                                \
            void *result =                                                     \
                context->acquire_operand(ss.operands[0], ss.qualifiers);       \
            if (!result) {                                                     \
                PTX_DEBUG_EMU("Failed to get operand address for op[%d]", 0);  \
                return false;                                                  \
            } else {                                                           \
                ss.operands[0].operand_phy_addr = result;                      \
            }                                                                  \
        }                                                                      \
        context->collect_operands(stmt, ss.operands, &(ss.qualifiers));        \
        auto iter = context->label2pc.find(ss.target);                         \
        assert(iter != context->label2pc.end());                               \
        context->operand_collected[1] = &(iter->second);                       \
        context->next_pc = context->pc + 1;                                    \
        return true;                                                           \
    }                                                                          \
    bool Name::operate(ThreadContext *context, StatementContext &stmt) {       \
        process_operation(context, &(context->operand_collected[0]),           \
                          stmt.qualifier);                                     \
        return true;                                                           \
    }                                                                          \
    bool Name::commit(ThreadContext *context, StatementContext &stmt) {        \
        const PredicatePrefix &ss = std::get<PredicatePrefix>(stmt.data);      \
        ss.operands[0].operand_phy_addr = nullptr;                             \
        stmt.state = InstructionState::COMMIT;                                 \
        context->trace_status(ptxsim::log_level::debug, "thread",              \
                              "Commit: NEXT_PC=%x ", context->next_pc);        \
        return true;                                                           \
    }

#define IMPLEMENT_ATOM_INSTR(Name)                                             \
    void Name::ExecPipe(ThreadContext *context, StatementContext &stmt) {      \
        const AtomInstr &ss = std::get<AtomInstr>(stmt.data);                  \
        context->next_pc = context->pc + 1;                                    \
        return;                                                                \
    }                                                                          \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    bool Name::operate(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    bool Name::commit(ThreadContext *context, StatementContext &stmt) {        \
        return true;                                                           \
    }

#define IMPLEMENT_WMMA_INSTR(Name)                                             \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    void Name::ExecPipe(ThreadContext *context, StatementContext &stmt) {      \
        const WmmaInstr &ss = std::get<WmmaInstr>(stmt.data);                  \
        context->next_pc = context->pc + 1;                                    \
        return;                                                                \
    }                                                                          \
    bool Name::operate(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    bool Name::commit(ThreadContext *context, StatementContext &stmt) {        \
        return true;                                                           \
    }

// CALL_INSTR
#define IMPLEMENT_CALL_INSTR(Name)                                             \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
        context->trace_status(ptxsim::log_level::debug, "thread",              \
                              "PC=%x " #Name " : %s", context->pc,             \
                              stmt.instructionText.c_str());                   \
        const CallInstr &ss = std::get<CallInstr>(stmt.data);                  \
        /* Pre-validate operand addresses */                                   \
        for (int i = 0; i < op_count; i++) {                                   \
            if (!ss.operands[i].operand_phy_addr) {                            \
                void *result =                                                 \
                    context->acquire_operand(ss.operands[i], ss.qualifiers);   \
                if (!result) {                                                 \
                    PTX_DEBUG_EMU("Failed to get operand address for op[%d]",  \
                                  i);                                          \
                    return false;                                              \
                } else {                                                       \
                    ss.operands[i].operand_phy_addr = result;                  \
                }                                                              \
            }                                                                  \
        }                                                                      \
        context->collect_operands(stmt, ss.operands, &(ss.qualifiers));        \
    }                                                                          \
    void Name::ExecPipe(ThreadContext *context, StatementContext &stmt) {      \
        const CallInstr &ss = std::get<CallInstr>(stmt.data);                  \
        if (ss.funcName == "printf" || ss.funcName == "_printf") {             \
            handlePrintf(context, stmt);                                       \
            context->next_pc = context->pc + 1;                                \
        } else {                                                               \
            INSTR_BASE::ExecPipe(context, stmt);                               \
        }                                                                      \
        return;                                                                \
    }                                                                          \
    bool Name::commit(ThreadContext *context, StatementContext &stmt) {        \
        const CallInstr &ss = std::get<CallInstr>(stmt.data);                  \
        ss.operands[0].operand_phy_addr = nullptr;                             \
        stmt.state = InstructionState::COMMIT;                                 \
        context->trace_status(ptxsim::log_level::debug, "thread",              \
                              "Commit: NEXT_PC=%x ", context->next_pc);        \
        return true;                                                           \
    }

#define IMPLEMENT_ASYNC_STORE(Name)
#define IMPLEMENT_ASYNC_REDUCE(Name)
#define IMPLEMENT_TCGEN_INSTR(Name)
#define IMPLEMENT_TENSORMAP_INSTR(Name)

#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    IMPLEMENT_##struct_kind(type_name)
#include "ptx_ir/ptx_op.def"
#undef X