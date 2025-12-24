#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/instruction_handlers_decl.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"

#define EXECUTE_OPERAND_REG(Name)                                              \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
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
    }                                                                          \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        return true; /* Nothing to do in execute for register declaration */   \
    }                                                                          \
    bool Name::commit(ThreadContext *context, StatementContext &stmt) {        \
        return true; /* Nothing to do in commit for register declaration */    \
    }                                                                          \
    void Name::execute_full(ThreadContext *context, StatementContext &stmt) {  \
        if (!prepare(context, stmt))                                           \
            return;                                                            \
        if (!execute(context, stmt))                                           \
            return;                                                            \
        commit(context, stmt);                                                 \
        context->pc++; /* Increment PC after successful execution */           \
    }                                                                          \
    void Name::process_operation(ThreadContext *context, void **operands,      \
                                 const std::vector<Qualifier> &qualifiers) {}

#define EXECUTE_OPERAND_CONST(Name)                                            \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        assert(false);                                                         \
        return true; /* Return true to satisfy bool return type */             \
    }                                                                          \
    void Name::process_operation(ThreadContext *context, void **operands,      \
                                 const std::vector<Qualifier> &qualifiers) {   \
        /* CONST operands are handled during parsing, no runtime operation */  \
    }

#define EXECUTE_OPERAND_MEMORY(Name)                                           \
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
    }                                                                          \
    void Name::process_operation(ThreadContext *context, void **operands,      \
                                 const std::vector<Qualifier> &qualifiers) {   \
        /* Memory operand setup is handled in execute, no runtime operation */ \
    }

// dollar TODO
#define EXECUTE_SIMPLE_NAME(Name)                                              \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    void Name::process_operation(ThreadContext *context, void **operands,      \
                                 const std::vector<Qualifier> &qualifiers) {   \
        /* Simple name operand requires no runtime operation */                \
    }

// PRAGMA TODO
#define EXECUTE_SIMPLE_STRING(Name)                                            \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    void Name::process_operation(ThreadContext *context, void **operands,      \
                                 const std::vector<Qualifier> &qualifiers) {   \
        /* String operand requires no runtime operation */                     \
    }

// RET TODO
#define EXECUTE_VOID_INSTR(Name)                                               \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    void Name::process_operation(ThreadContext *context, void **operands,      \
                                 const std::vector<Qualifier> &qualifiers) {   \
        /* Void instruction requires no runtime operation */                   \
    }

// BRA
// -1 because pc will be incremented after this instruction
#define EXECUTE_BRANCH(Name)                                                   \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        auto iter = context->label2pc.find(ss->braTarget);                     \
        assert(iter != context->label2pc.end());                               \
        void *op[0];                                                           \
        op[0] = &(iter->second);                                               \
        process_operation(context, op, ss->qualifier);                         \
        return true;                                                           \
    }

#define EXECUTE_PREDICATE_PREFIX(Name)                                         \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }                                                                          \
    void Name::process_operation(ThreadContext *context, void **operands,      \
                                 const std::vector<Qualifier> &qualifiers) {   \
        /* Predicate prefix requires no runtime operation */                   \
    }

#define EXECUTE_GENERIC_INSTR(Name, OpCount)                                   \
    bool Name::prepare(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        /* Pre-validate operand addresses */                                   \
        for (int i = 0; i < OpCount; i++) {                                    \
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
        void *op[OpCount];                                                     \
        for (int i = 0; i < OpCount; i++) {                                    \
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

#define EXECUTE_ATOM_INSTR(Name, OpCount)                                      \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        void *op[OpCount];                                                     \
        for (int i = 0; i < OpCount; i++) {                                    \
            op[i] = context->get_operand_addr(ss->op[i], ss->qualifier);       \
        }                                                                      \
        process_operation(context, op, ss->qualifier);                         \
        return true;                                                           \
    }

#define EXECUTE_WMMA_INSTR(Name, OpCount)                                      \
    bool Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        return true;                                                           \
    }

#define EXECUTE_BARRIER(Name)                                                  \
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
    void Name::process_operation(ThreadContext *context, void **operands,      \
                                 const std::vector<Qualifier> &qualifiers) {   \
        /* Barrier operation handled in execute method */                      \
    }
// =============================================================================
// 5. 主分发宏
// =============================================================================
// Overloads for kinds that don't use OpCount
#define DISPATCH_EXECUTE_OPERAND_REG(Name, _) EXECUTE_OPERAND_REG(Name)
#define DISPATCH_EXECUTE_OPERAND_CONST(Name, _) EXECUTE_OPERAND_CONST(Name)
#define DISPATCH_EXECUTE_OPERAND_MEMORY(Name, _) EXECUTE_OPERAND_MEMORY(Name)
#define DISPATCH_EXECUTE_SIMPLE_NAME(Name, _) EXECUTE_SIMPLE_NAME(Name)
#define DISPATCH_EXECUTE_SIMPLE_STRING(Name, _) EXECUTE_SIMPLE_STRING(Name)
#define DISPATCH_EXECUTE_VOID_INSTR(Name, _) EXECUTE_VOID_INSTR(Name)
#define DISPATCH_EXECUTE_BRANCH(Name, _) EXECUTE_BRANCH(Name)
#define DISPATCH_EXECUTE_BARRIER(Name, _) EXECUTE_BARRIER(Name)
#define DISPATCH_EXECUTE_PREDICATE_PREFIX(Name, _)                             \
    EXECUTE_PREDICATE_PREFIX(Name)
#define DISPATCH_EXECUTE_GENERIC_INSTR(Name, cnt)                              \
    EXECUTE_GENERIC_INSTR(Name, cnt)
#define DISPATCH_EXECUTE_WMMA_INSTR(Name, cnt) EXECUTE_WMMA_INSTR(Name, cnt)
#define DISPATCH_EXECUTE_ATOM_INSTR(Name, cnt) EXECUTE_ATOM_INSTR(Name, cnt)

// =============================================================================
// 6. 生成所有execute函数
// =============================================================================

#define DISPATCH_EXECUTE(struct_kind, type_name, op_count)                     \
    DISPATCH_EXECUTE_##struct_kind(type_name, op_count)

#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    DISPATCH_EXECUTE(struct_kind, type_name, op_count)
#include "ptx_ir/ptx_op.def"
#undef X