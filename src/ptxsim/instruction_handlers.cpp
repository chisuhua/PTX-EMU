#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h" // 包含 StatementContext 定义
#include "ptxsim/instruction_handlers_decl.h"
#include "ptxsim/thread_context.h" // 假设存在

#define EXECUTE_OPERAND_REG(Name)                                              \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = static_cast<StatementContext::Name *>(stmt.statement);       \
        for (int i = 0; i < ss->regNum; i++) {                                 \
            PtxInterpreter::Reg *r = new PtxInterpreter::Reg();                \
            r->byteNum = TypeUtils::get_bytes(ss->regDataType);                \
            r->elementNum = 1;                                                 \
            r->name = ss->regName + std::to_string(i);                         \
            r->regType = ss->regDataType.back();                               \
            r->addr = new char[r->byteNum];                                    \
            memset(r->addr, 0, r->byteNum);                                    \
            context->name2Reg[r->name] = r;                                    \
            std::cout << "Registered register: " << r->name << std::endl;      \
        }                                                                      \
    }

#define EXECUTE_OPERAND_CONST(Name)                                            \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        assert(false);                                                         \
    }

#define EXECUTE_OPERAND_MEMORY(Name)                                           \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        PtxInterpreter::Symtable *s = new PtxInterpreter::Symtable();          \
        s->byteNum = TypeUtils::get_bytes(ss->dataType) * ss->size;            \
        s->elementNum = ss->size;                                              \
        s->name = ss->name;                                                    \
        s->symType = ss->dataType.back();                                      \
        s->val = (uint64_t)(new char[s->byteNum]);                             \
        memset((void *)s->val, 0, s->byteNum);                                 \
        (*context->name2Share)[s->name] = s;                                   \
        context->name2Sym[s->name] = s;                                        \
    }

// dollar TODO
#define EXECUTE_SIMPLE_NAME(Name)                                              \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {}

// PRAGMA TODO
#define EXECUTE_SIMPLE_STRING(Name)                                            \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {}

// RET TODO
#define EXECUTE_VOID_INSTR(Name)                                               \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {}

// BRA
// -1 because pc will be incremented after this instruction
#define EXECUTE_BRANCH(Name)                                                   \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        auto iter = context->label2pc.find(ss->braTarget);                     \
        assert(iter != context->label2pc.end());                               \
        void *op[0];                                                           \
        op[0] = &(iter->second);                                               \
        process_operation(context, op, ss->qualifier);                         \
    }

// BAR
// #ifdef DEBUGINTE sync_thread = 1;
// #endif
//         #ifdef
// LOGINTE if (IFLOG()) {                                      \
//                 std::cout << "INTE: Thread(" << context->ThreadIdx.x << "," \
//                           << context->ThreadIdx.y << "," \
//                           << context->ThreadIdx.z << ") in Block(" \
//                           << context->BlockIdx.x << "," <<
//                           context->BlockIdx.y \
//                           << "," << context->BlockIdx.z << ") bar.sync" \
//                           << std::endl; \
//             } \
//             #endif \

#define EXECUTE_PREDICATE_PREFIX(Name)                                         \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {}

#define EXECUTE_GENERIC_INSTR(Name, OpCount)                                   \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        void *op[OpCount];                                                     \
        for (int i = 0; i < OpCount; i++) {                                    \
            op[i] = context->get_operand_addr(ss->op[i], ss->qualifier);       \
        }                                                                      \
        process_operation(context, op, ss->qualifier);                         \
    }

#define EXECUTE_ATOM_INSTR(Name, OpCount)                                      \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
        void *op[OpCount];                                                     \
        for (int i = 0; i < OpCount; i++) {                                    \
            op[i] = context->get_operand_addr(ss->op[i], ss->qualifier);       \
        }                                                                      \
        process_operation(context, op, ss->qualifier);                         \
    }

#define EXECUTE_WMMA_INSTR(Name, OpCount)                                      \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {}

#define EXECUTE_BARRIER(Name)                                                  \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto ss = (StatementContext::Name *)stmt.statement;                    \
                                                                               \
        if (ss->barType == "sync") {                                           \
            context->state = BAR_SYNC;                                         \
        } else {                                                               \
            assert(false && "Unsupported barrier type");                       \
        }                                                                      \
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