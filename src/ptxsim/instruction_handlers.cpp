#include "instruction_handlers_decl.h"
#include "statement_context.h" // 包含 StatementContext 定义
#include "thread_context.h"    // 假设存在

// Helper: get raw pointer from OperandContext
// Adjust this based on your OperandContext implementation
static void *getOperandPtr(OperandContext &op) {
    return op.getRawPtr(); // You must implement getRawPtr() or equivalent
}

// Helper macros: implement execute() based on op_count
#define IMPLEMENT_EXECUTE_0OP(Name)                                            \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto &s = static_cast<StatementContext::Name &>(stmt.statement);       \
        process_operation(context, s.qualifier);                               \
    }

#define IMPLEMENT_EXECUTE_1OP(Name)                                            \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto &s = static_cast<StatementContext::Name &>(stmt.statement);       \
        void *op0 = getOperandPtr(s.op[0]);                                    \
        process_operation(context, op0, s.qualifier);                          \
    }

#define IMPLEMENT_EXECUTE_2OP(Name)                                            \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto &s = static_cast<StatementContext::Name &>(stmt.statement);       \
        void *op0 = getOperandPtr(s.op[0]);                                    \
        void *op1 = getOperandPtr(s.op[1]);                                    \
        process_operation(context, op0, op1, s.qualifier);                     \
    }

#define IMPLEMENT_EXECUTE_3OP(Name)                                            \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto &s = static_cast<StatementContext::Name &>(stmt.statement);       \
        void *op0 = getOperandPtr(s.op[0]);                                    \
        void *op1 = getOperandPtr(s.op[1]);                                    \
        void *op2 = getOperandPtr(s.op[2]);                                    \
        process_operation(context, op0, op1, op2, s.qualifier);                \
    }

#define IMPLEMENT_EXECUTE_4OP(Name)                                            \
    void Name::execute(ThreadContext *context, StatementContext &stmt) {       \
        auto &s = static_cast<StatementContext::Name &>(stmt.statement);       \
        void *op0 = getOperandPtr(s.op[0]);                                    \
        void *op1 = getOperandPtr(s.op[1]);                                    \
        void *op2 = getOperandPtr(s.op[2]);                                    \
        void *op3 = getOperandPtr(s.op[3]);                                    \
        process_operation(context, op0, op1, op2, op3, s.qualifier);           \
    }

// Generate all execute() implementations
#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    IMPLEMENT_EXECUTE_##op_count##OP(type_name)
#include "ptx_ir/ptx_op.def"
#undef X