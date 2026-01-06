#include "ptxsim/instruction_handlers.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include "utils/logger.h"

// for (int i = 0; i < ss->regNum; i++) {                                 \
        //     std::string reg_name = ss->regName + std::to_string(i);            \
        //     size_t reg_size = getBytes(ss->regDataType);                       \
        //     /* Create register in RegisterManager */                           \
        //     if (!context->register_manager.create_register(reg_name,           \
        //                                                    reg_size)) {        \
        //         PTX_DEBUG_EMU("Failed to create register: %s with size %zu",   \
        //                       reg_name.c_str(), reg_size);                     \
        //         return;                                                        \
        //     }                                                                  \
        //     PTX_DEBUG_EMU("Registered register: %s with size %zu",             \
        //                   reg_name.c_str(), reg_size);                         \
        // }
#define IMPLEMENT_OPERAND_REG(Name)                                            \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = static_cast<StatementContext::Name *>(stmt.statement);       \
        context->next_pc = context->pc + 1;                                    \
        return;                                                                \
    }

#define IMPLEMENT_OPERAND_CONST(Name)                                          \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        assert(false);                                                         \
        context->next_pc = context->pc + 1;                                    \
        return; /* Return true to satisfy bool return type */                  \
    }

#define IMPLEMENT_OPERAND_MEMORY(Name)                                         \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        Symtable *s = new Symtable();                                          \
        s->byteNum = getBytes(ss->dataType) * ss->size;                        \
        s->elementNum = ss->size;                                              \
        s->name = ss->name;                                                    \
        s->symType = ss->dataType.back();                                      \
        s->val = (uint64_t)(new char[s->byteNum]);                             \
        memset((void *)s->val, 0, s->byteNum);                                 \
        (*context->name2Sym)[s->name] = s;                                     \
        context->next_pc = context->pc + 1;                                    \
        return;                                                                \
    }

// dollar TODO
#define IMPLEMENT_SIMPLE_NAME(Name)                                            \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        context->next_pc = context->pc + 1;                                    \
        return;                                                                \
    }

// PRAGMA TODO
#define IMPLEMENT_SIMPLE_STRING(Name)                                          \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        context->next_pc = context->pc + 1;                                    \
        return;                                                                \
    }

// RET TODO
#define IMPLEMENT_VOID_INSTR(Name)                                             \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        context->state = EXIT;                                                 \
        return;                                                                \
    }

#define IMPLEMENT_PREDICATE_PREFIX(Name)                                       \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        context->next_pc = context->pc + 1;                                    \
        return;                                                                \
    }

// BRA
// -1 because pc will be incremented after this instruction
#define IMPLEMENT_BRANCH(Name)                                                 \
    bool Name::operate(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        auto iter = context->label2pc.find(ss->braTarget);                     \
        assert(iter != context->label2pc.end());                               \
        void *op[0];                                                           \
        op[0] = &(iter->second);                                               \
        process_operation(context, op, ss->qualifier);                         \
        return true;                                                           \
    }

#define IMPLEMENT_BARRIER(Name)                                                \
    bool Name::operate(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
                                                                               \
        if (ss->barType == "sync") {                                           \
            context->state = BAR_SYNC;                                         \
        } else {                                                               \
            assert(false && "Unsupported barrier type");                       \
        }                                                                      \
        return true;                                                           \
    }                                                                          \
    void Name::process_operation(ThreadContext *context, void **operands,      \
                                 const std::vector<Qualifier> &qualifiers) {   \
        /* Barrier operation handled in execute method */                      \
    }

#define IMPLEMENT_GENERIC_INSTR(Name)                                          \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
        PTX_DEBUG_EMU("run to prepare" #Name);                                 \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        /* Pre-validate operand addresses */                                   \
        for (int i = 0; i < op_count; i++) {                                   \
            if (!ss->operands[i].operand_phy_addr) {                           \
                void *result =                                                 \
                    context->acquire_operand(ss->operands[i], ss->qualifier);  \
                if (!result) {                                                 \
                    PTX_DEBUG_EMU("Failed to get operand address for op[%d]",  \
                                  i);                                          \
                    return false;                                              \
                } else {                                                       \
                    ss->operands[i].operand_phy_addr = result;                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
        context->collect_operands(stmt, ss->operands, &(ss->qualifier));       \
        context->next_pc = context->pc + 1;                                    \
        return true;                                                           \
    }                                                                          \
    bool Name::commit(ThreadContext *context, StatementContext &stmt) {        \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        context->commit_operand(stmt, ss->operands[0], ss->qualifier);         \
        for (int i = 0; i < op_count; i++) {                                   \
            ss->operands[i].operand_phy_addr = nullptr;                        \
        }                                                                      \
        stmt.state = InstructionState::COMMIT;                                 \
        return true; /* Typically no commit work needed */                     \
    }

#define IMPLEMENT_ATOM_INSTR(Name)                                             \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    bool Name::operate(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        void *op[op_count];                                                    \
        for (int i = 0; i < op_count; i++) {                                   \
            op[i] = context->acquire_operand(ss->operands[i], ss->qualifier);  \
        }                                                                      \
        process_operation(context, op, ss->qualifier);                         \
        return true;                                                           \
    }                                                                          \
    bool Name::commit(ThreadContext *context, StatementContext &stmt) {        \
        return true;                                                           \
    }

#define IMPLEMENT_WMMA_INSTR(Name)                                             \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    bool Name::operate(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    bool Name::commit(ThreadContext *context, StatementContext &stmt) {        \
        return true;                                                           \
    }

#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    IMPLEMENT_##struct_kind(type_name)
#include "ptx_ir/ptx_op.def"
#undef X

// bool Name::operate(ThreadContext *context, StatementContext &stmt) {       \
    //     auto ss = (StatementContext::Name *)stmt.statement;                    \
    //     void *op_phy_addr[op_count];                                           \
    //     for (int i = 0; i < op_count; i++) {                                   \
    //         op_phy_addr[i] = ss->operands[i].operand_addr;                     \
    //     }                                                                      \
    //     process_operation(context, op_phy_addr, ss->qualifier);                \
    //     return true;                                                           \
    // }                                                                          \
