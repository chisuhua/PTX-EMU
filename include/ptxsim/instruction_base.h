// InstructionHandlers_decl.h
#ifndef INSTRUCTION_HANDLE_H
#define INSTRUCTION_HANDLE_H
#include "../ptx_ir/statement_context.h"
#include <memory>
#include <vector>

class ThreadContext;

// 基础指令处理器接口
class InstructionHandler {
public:
    virtual ~InstructionHandler() = default;

    // 分阶段执行接口
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool execute(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool commit(ThreadContext *context, StatementContext &stmt) = 0;
    // 保持向后兼容的execute接口 - 使用不同名称避免重载冲突
    virtual void execute_full(ThreadContext *context, StatementContext &stmt);
};

// OPERAND_REG 类型的基类处理器
class OPERAND_REG : public InstructionHandler {
public:
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
    virtual bool commit(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
};

// OPERAND_CONST 类型的基类处理器
class OPERAND_CONST : public InstructionHandler {
public:
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
    virtual bool commit(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
};

// OPERAND_MEMORY 类型的基类处理器
class OPERAND_MEMORY : public InstructionHandler {
public:
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
    virtual bool commit(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
};

// SIMPLE_NAME 类型的基类处理器
class SIMPLE_NAME : public InstructionHandler {
public:
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
    virtual bool commit(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
};

// SIMPLE_STRING 类型的基类处理器
class SIMPLE_STRING : public InstructionHandler {
public:
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
    virtual bool commit(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
};

// VOID_INSTR 类型的基类处理器
class VOID_INSTR : public InstructionHandler {
public:
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
    virtual bool commit(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
};

// PREDICATE_PREFIX 类型的基类处理器
class PREDICATE_PREFIX : public InstructionHandler {
public:
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
    virtual bool commit(ThreadContext *context, StatementContext &stmt) {
        return true;
    };
};

// BRANCH 类型的基类处理器
class BRANCH : public InstructionHandler {
public:
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool execute(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool commit(ThreadContext *context, StatementContext &stmt) = 0;
    virtual void
    process_operation(ThreadContext *context, void **operands,
                      const std::vector<Qualifier> &qualifiers) = 0;
};

// GENERIC_INSTR 类型的基类处理器
class GENERIC_INSTR : public InstructionHandler {
public:
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool execute(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool commit(ThreadContext *context, StatementContext &stmt) = 0;
    virtual void
    process_operation(ThreadContext *context, void **operands,
                      const std::vector<Qualifier> &qualifiers) = 0;
};

// ATOM_INSTR 类型的基类处理器
class ATOM_INSTR : public InstructionHandler {
public:
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool execute(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool commit(ThreadContext *context, StatementContext &stmt) = 0;
    virtual void
    process_operation(ThreadContext *context, void **operands,
                      const std::vector<Qualifier> &qualifiers) = 0;
};

// WMMA_INSTR 类型的基类处理器
class WMMA_INSTR : public InstructionHandler {
public:
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool execute(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool commit(ThreadContext *context, StatementContext &stmt) = 0;
    virtual void
    process_operation(ThreadContext *context, void **operands,
                      const std::vector<Qualifier> &qualifiers) = 0;
};

// BARRIER 类型的基类处理器
class BARRIER : public InstructionHandler {
public:
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool execute(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool commit(ThreadContext *context, StatementContext &stmt) = 0;
    virtual void
    process_operation(ThreadContext *context, void **operands,
                      const std::vector<Qualifier> &qualifiers) = 0;
};

#endif