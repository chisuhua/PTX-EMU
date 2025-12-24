#include "ptxsim/instruction_handlers.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"

#define IMPLEMENT_OPERAND_REG(Name)                                            \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = static_cast<StatementContext::Name *>(stmt.statement);       \
        for (int i = 0; i < ss->regNum; i++) {                                 \
            std::string reg_name = ss->regName + std::to_string(i);            \
            size_t reg_size = getBytes(ss->regDataType);                       \
            /* Create register in RegisterManager */                           \
            if (!context->register_manager.create_register(reg_name,           \
                                                           reg_size)) {        \
                PTX_DEBUG_EMU("Failed to create register: %s with size %zu",   \
                              reg_name.c_str(), reg_size);                     \
                return false;                                                  \
            }                                                                  \
            PTX_DEBUG_EMU("Registered register: %s with size %zu",             \
                          reg_name.c_str(), reg_size);                         \
        }                                                                      \
        return true;                                                           \
    }

#define IMPLEMENT_OPERAND_CONST(Name)                                          \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        assert(false);                                                         \
        return true; /* Return true to satisfy bool return type */             \
    }

#define IMPLEMENT_OPERAND_MEMORY(Name)                                         \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        PtxInterpreter::Symtable *s = new PtxInterpreter::Symtable();          \
        s->byteNum = getBytes(ss->dataType) * ss->size;                        \
        s->elementNum = ss->size;                                              \
        s->name = ss->name;                                                    \
        s->symType = ss->dataType.back();                                      \
        s->val = (uint64_t)(new char[s->byteNum]);                             \
        memset((void *)s->val, 0, s->byteNum);                                 \
        (*context->name2Share)[s->name] = s;                                   \
        context->name2Sym[s->name] = s;                                        \
        return true;                                                           \
    }

// dollar TODO
#define IMPLEMENT_SIMPLE_NAME(Name)                                            \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }

// PRAGMA TODO
#define IMPLEMENT_SIMPLE_STRING(Name)                                          \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }

// RET TODO
#define IMPLEMENT_VOID_INSTR(Name)                                             \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }

#define IMPLEMENT_PREDICATE_PREFIX(Name)                                       \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }

// BRA
// -1 because pc will be incremented after this instruction
#define IMPLEMENT_BRANCH(Name)                                                 \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
        return true; /* Typically no commit work needed */                     \
    }                                                                          \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        auto iter = context->label2pc.find(ss->braTarget);                     \
        assert(iter != context->label2pc.end());                               \
        void *op[0];                                                           \
        op[0] = &(iter->second);                                               \
        process_operation(context, op, ss->qualifier);                         \
        return true;                                                           \
    }                                                                          \
    bool Name::commit(ThreadContext *context, StatementContext &stmt) {        \
        return true; /* Typically no commit work needed */                     \
    }

#define IMPLEMENT_GENERIC_INSTR(Name)                                          \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        /* Pre-validate operand addresses */                                   \
        for (int i = 0; i < op_count; i++) {                                   \
            void *op_addr =                                                    \
                context->get_operand_addr(ss->op[i], ss->qualifier);           \
            if (!op_addr) {                                                    \
                PTX_DEBUG_EMU("Failed to get operand address for op[%d]", i);  \
                return false;                                                  \
            }                                                                  \
        }                                                                      \
        return true;                                                           \
    }                                                                          \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        void *op[op_count];                                                    \
        for (int i = 0; i < op_count; i++) {                                   \
            op[i] = context->get_operand_addr(ss->op[i], ss->qualifier);       \
        }                                                                      \
        process_operation(context, op, ss->qualifier);                         \
        return true;                                                           \
    }                                                                          \
    bool Name::commit(ThreadContext *context, StatementContext &stmt) {        \
        return true; /* Typically no commit work needed */                     \
    }                                                                          \
    void Name::execute_full(ThreadContext *context, StatementContext &stmt) {  \
        if (!prepare(context, stmt))                                           \
            return;                                                            \
        if (!execute(context, stmt))                                           \
            return;                                                            \
        commit(context, stmt);                                                 \
        context->pc++; /* Increment PC after successful execution */           \
    }

#define IMPLEMENT_ATOM_INSTR(Name)                                             \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        void *op[op_count];                                                    \
        for (int i = 0; i < op_count; i++) {                                   \
            op[i] = context->get_operand_addr(ss->op[i], ss->qualifier);       \
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
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    bool Name::commit(ThreadContext *context, StatementContext &stmt) {        \
        return true;                                                           \
    }

#define IMPLEMENT_BARRIER(Name)                                                \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
                                                                               \
        if (ss->barType == "sync") {                                           \
            context->state = BAR_SYNC;                                         \
        } else {                                                               \
            assert(false && "Unsupported barrier type");                       \
        }                                                                      \
        return true;                                                           \
    }                                                                          \
    bool Name::commit(ThreadContext *context, StatementContext &stmt) {        \
        return true;                                                           \
    }                                                                          \
    void Name::process_operation(ThreadContext *context, void **operands,      \
                                 const std::vector<Qualifier> &qualifiers) {   \
        /* Barrier operation handled in execute method */                      \
    }

#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    IMPLEMENT_##struct_kind(type_name)
#include "ptx_ir/ptx_op.def"
#undef X