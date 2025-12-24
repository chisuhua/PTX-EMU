// instruction_handlers.h
#ifndef PTXSIM_INSTRUCTION_HANDLERS_DECL_H
#define PTXSIM_INSTRUCTION_HANDLERS_DECL_H

#include "instruction_base.h"

class ThreadContext;
class StatementContext;

#define DECLARE_OPERAND_REG(Name, _)                                           \
    class Name : public OPERAND_REG {                                          \
    public:                                                                    \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool execute(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_OPERAND_CONST(Name, _)                                         \
    class Name : public OPERAND_CONST {                                        \
    public:                                                                    \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool execute(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_OPERAND_MEMORY(Name, _)                                        \
    class Name : public OPERAND_MEMORY {                                       \
    public:                                                                    \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool execute(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_SIMPLE_NAME(Name, _)                                           \
    class Name : public SIMPLE_NAME {                                          \
    public:                                                                    \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool execute(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_SIMPLE_STRING(Name, _)                                         \
    class Name : public SIMPLE_STRING {                                        \
    public:                                                                    \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool execute(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_VOID_INSTR(Name, _)                                            \
    class Name : public VOID_INSTR {                                           \
    public:                                                                    \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool execute(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_BRANCH(Name, _)                                                \
    class Name : public BRANCH {                                               \
    public:                                                                    \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool execute(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_BARRIER(Name, _)                                               \
    class Name : public BARRIER {                                              \
    public:                                                                    \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool execute(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_PREDICATE_PREFIX(Name, _)                                      \
    class Name : public PREDICATE_PREFIX {                                     \
    public:                                                                    \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool execute(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_GENERIC_INSTR(Name, OpCount)                                   \
    class Name : public GENERIC_INSTR {                                        \
    public:                                                                    \
        static constexpr int op_count = OpCount;                               \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool execute(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void execute_full(ThreadContext *context,                              \
                          StatementContext &stmt) override;                    \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_WMMA_INSTR(Name, OpCount)                                      \
    class Name : public WMMA_INSTR {                                           \
    public:                                                                    \
        static constexpr int op_count = OpCount;                               \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool execute(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_ATOM_INSTR(Name, OpCount)                                      \
    class Name : public ATOM_INSTR {                                           \
    public:                                                                    \
        static constexpr int op_count = OpCount;                               \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool execute(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    DECLARE_##struct_kind(type_name, op_count)
#include "ptx_ir/ptx_op.def"
#undef X

#endif // PTXSIM_INSTRUCTION_HANDLERS_DECL_H