// InstructionHandlers_decl.h
#ifndef INSTRUCTION_HANDLE_DECL_H
#define INSTRUCTION_HANDLE_DECL_H

#include "instruction_handler.h"
#include <vector>

// Helper macro: generate class declaration based on op_count
#define DEFINE_HANDLER_0OP(Name)                                               \
    class Name : public InstructionHandler {                                   \
    public:                                                                    \
        void execute(ThreadContext *context, StatementContext &stmt) override; \
                                                                               \
    protected:                                                                 \
        virtual void process_operation(ThreadContext *context,                 \
                                       std::vector<Qualifier> &qualifiers);    \
    };

#define DEFINE_HANDLER_1OP(Name)                                               \
    class Name : public InstructionHandler {                                   \
    public:                                                                    \
        void execute(ThreadContext *context, StatementContext &stmt) override; \
                                                                               \
    protected:                                                                 \
        virtual void process_operation(ThreadContext *context, void *op0,      \
                                       std::vector<Qualifier> &qualifiers);    \
    };

#define DEFINE_HANDLER_2OP(Name)                                               \
    class Name : public InstructionHandler {                                   \
    public:                                                                    \
        void execute(ThreadContext *context, StatementContext &stmt) override; \
                                                                               \
    protected:                                                                 \
        virtual void process_operation(ThreadContext *context, void *op0,      \
                                       void *op1,                              \
                                       std::vector<Qualifier> &qualifiers);    \
    };

#define DEFINE_HANDLER_3OP(Name)                                               \
    class Name : public InstructionHandler {                                   \
    public:                                                                    \
        void execute(ThreadContext *context, StatementContext &stmt) override; \
                                                                               \
    protected:                                                                 \
        virtual void process_operation(ThreadContext *context, void *op0,      \
                                       void *op1, void *op2,                   \
                                       std::vector<Qualifier> &qualifiers);    \
    };

#define DEFINE_HANDLER_4OP(Name)                                               \
    class Name : public InstructionHandler {                                   \
    public:                                                                    \
        void execute(ThreadContext *context, StatementContext &stmt) override; \
                                                                               \
    protected:                                                                 \
        virtual void process_operation(ThreadContext *context, void *op0,      \
                                       void *op1, void *op2, void *op3,        \
                                       std::vector<Qualifier> &qualifiers);    \
    };

// Generate all handler class declarations
#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    DEFINE_HANDLER_##op_count##OP(type_name)
#include "ptx_ir/ptx_op.def"
#undef X

#endif