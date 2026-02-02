// instruction_handlers.h
#ifndef PTXSIM_INSTRUCTION_HANDLERS_DECL_H
#define PTXSIM_INSTRUCTION_HANDLERS_DECL_H

#include "instruction_base.h"

class ThreadContext;
class StatementContext;

#define DECLARE_ABI_DIRECTIVE(Name, _)                                         \
    class Name : public ABI_DIRECTIVE {                                        \
    public:                                                                    \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
    };

#define DECLARE_OPERAND_REG(Name, _)                                           \
    class Name : public OPERAND_REG {                                          \
    public:                                                                    \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
    };

#define DECLARE_OPERAND_CONST(Name, _)                                         \
    class Name : public OPERAND_CONST {                                        \
    public:                                                                    \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
    };

#define DECLARE_OPERAND_MEMORY(Name, _)                                        \
    class Name : public OPERAND_MEMORY {                                       \
    public:                                                                    \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
    };

#define DECLARE_SIMPLE_NAME(Name, _)                                           \
    class Name : public SIMPLE_NAME {                                          \
    public:                                                                    \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
    };

#define DECLARE_SIMPLE_STRING(Name, _)                                         \
    class Name : public SIMPLE_STRING {                                        \
    public:                                                                    \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
    };

#define DECLARE_VOID_INSTR(Name, _)                                            \
    class Name : public VOID_INSTR {                                           \
    public:                                                                    \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
        void process_operation(ThreadContext *context);                        \
    };

#define DECLARE_BRANCH(Name, _)                                                \
    class Name : public BRANCH {                                               \
    public:                                                                    \
        bool operate(ThreadContext *context, StatementContext &stmt) override; \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_BARRIER(Name, _)                                               \
    class Name : public BARRIER {                                              \
    public:                                                                    \
        bool operate(ThreadContext *context, StatementContext &stmt) override; \
        void process_operation(ThreadContext *context, int barId,              \
                               const std::vector<Qualifier> &qualifiers);      \
    };

#define DECLARE_GENERIC_INSTR(Name, OpCount)                                   \
    class Name : public GENERIC_INSTR {                                        \
    public:                                                                    \
        static constexpr int op_count = OpCount;                               \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_PREDICATE_PREFIX(Name, OpCount)                                \
    class Name : public PREDICATE_PREFIX {                                     \
    public:                                                                    \
        static constexpr int op_count = OpCount;                               \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool operate(ThreadContext *context, StatementContext &stmt) override; \
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
        bool operate(ThreadContext *context, StatementContext &stmt) override; \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_WMMA_INSTR(Name, OpCount)                                      \
    class Name : public WMMA_INSTR {                                           \
    public:                                                                    \
        static constexpr int op_count = OpCount;                               \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool operate(ThreadContext *context, StatementContext &stmt) override; \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_ASYNC_STORE(Name, OpCount)                                     \
    class Name : public ASYNC_STORE {                                          \
    public:                                                                    \
        static constexpr int op_count = OpCount;                               \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool operate(ThreadContext *context, StatementContext &stmt) override; \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_ASYNC_REDUCE(Name, OpCount)                                    \
    class Name : public ASYNC_REDUCE {                                         \
    public:                                                                    \
        static constexpr int op_count = OpCount;                               \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool operate(ThreadContext *context, StatementContext &stmt) override; \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_TCGEN_INSTR(Name, OpCount)                                     \
    class Name : public TCGEN_INSTR {                                          \
    public:                                                                    \
        static constexpr int op_count = OpCount;                               \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool operate(ThreadContext *context, StatementContext &stmt) override; \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

#define DECLARE_TENSORMAP_INSTR(Name, OpCount)                                 \
    class Name : public TENSORMAP_INSTR {                                      \
    public:                                                                    \
        static constexpr int op_count = OpCount;                               \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool operate(ThreadContext *context, StatementContext &stmt) override; \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
    };

// 专门为 CALL 指令添加声明宏
#define DECLARE_CALL_INSTR(Name, OpCount)                                      \
    class Name : public GENERIC_INSTR {                                        \
    public:                                                                    \
        static constexpr int op_count = OpCount;                               \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void ExecPipe(ThreadContext *context,                                  \
                      StatementContext &stmt) override;                        \
        void handlePrintf(ThreadContext *context, StatementContext &stmt);     \
        void                                                                   \
        process_operation(ThreadContext *context, void **operands,             \
                          const std::vector<Qualifier> &qualifiers) override;  \
        void parseAndPrintFormat(ThreadContext *context,                       \
                                 const std::string &format,                    \
                                 const std::vector<void *> &args);             \
    };

#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    DECLARE_##struct_kind(type_name, op_count)
#include "ptx_ir/ptx_op.def"
#undef X

#endif // PTXSIM_INSTRUCTION_HANDLERS_DECL_H