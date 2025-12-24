// instruction_handlers_decl.h
#ifndef PTXSIM_INSTRUCTION_HANDLERS_DECL_H
#define PTXSIM_INSTRUCTION_HANDLERS_DECL_H

#include "instruction_handler.h"

class ThreadContext;
class StatementContext;

// 为每种指令类型定义处理器类
// 这些类将实现InstructionHandler接口

// 通用宏定义
#define DECLARE_INSTRUCTION_HANDLER(name)                                      \
    class name : public InstructionHandler {                                   \
    public:                                                                    \
        bool prepare(ThreadContext *context, StatementContext &stmt) override; \
        bool execute(ThreadContext *context, StatementContext &stmt) override; \
        bool commit(ThreadContext *context, StatementContext &stmt) override;  \
        void execute_full(ThreadContext *context,                              \
                          StatementContext &stmt) override;                    \
        void process_operation(ThreadContext *context, void **operands,        \
                               const std::vector<Qualifier> &qualifiers) override; \
    };

// 包含所有指令处理器的声明
#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    DECLARE_INSTRUCTION_HANDLER(type_name)
#include "../ptx_ir/ptx_op.def"
#undef X

#endif // PTXSIM_INSTRUCTION_HANDLERS_DECL_H