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
    virtual void execute(ThreadContext *context, StatementContext &stmt) = 0;
};

// OPERAND_REG 类型的基类处理器
class OPERAND_REG : public InstructionHandler {
public:
    // virtual void execute(ThreadContext *context, StatementContext &stmt);
};

// OPERAND_CONST 类型的基类处理器
class OPERAND_CONST : public InstructionHandler {
public:
    // virtual void execute(ThreadContext *context, StatementContext &stmt);
};

// OPERAND_MEMORY 类型的基类处理器
class OPERAND_MEMORY : public InstructionHandler {
public:
    // virtual void execute(ThreadContext *context, StatementContext &stmt);
};

// SIMPLE_NAME 类型的基类处理器
class SIMPLE_NAME : public InstructionHandler {
public:
    // virtual void execute(ThreadContext *context, StatementContext &stmt);
};

// SIMPLE_STRING 类型的基类处理器
class SIMPLE_STRING : public InstructionHandler {
public:
    // virtual void execute(ThreadContext *context, StatementContext &stmt);
};

// VOID_INSTR 类型的基类处理器
class VOID_INSTR : public InstructionHandler {
public:
    // virtual void execute(ThreadContext *context, StatementContext &stmt);
};

// PREDICATE_PREFIX 类型的基类处理器
class PREDICATE_PREFIX : public InstructionHandler {
public:
    // virtual void execute(ThreadContext *context, StatementContext &stmt);
};

class INSTR_BASE : public InstructionHandler {
public:
    virtual void execute(ThreadContext *context, StatementContext &stmt);
    virtual bool prepare(ThreadContext *context, StatementContext &stmt);
    virtual bool commit(ThreadContext *context, StatementContext &stmt);

    virtual bool operate(ThreadContext *context, StatementContext &stmt) = 0;
    virtual void
    process_operation(ThreadContext *context, void **operands,
                      const std::vector<Qualifier> &qualifiers) = 0;
};

// BRANCH 类型的基类处理器
class BRANCH : public INSTR_BASE {
public:
};

// GENERIC_INSTR 类型的基类处理器
class GENERIC_INSTR : public INSTR_BASE {
public:
};

// ATOM_INSTR 类型的基类处理器
class ATOM_INSTR : public INSTR_BASE {
public:
};

// WMMA_INSTR 类型的基类处理器
class WMMA_INSTR : public INSTR_BASE {
public:
};

// BARRIER 类型的基类处理器
class BARRIER : public INSTR_BASE {
public:
};

#endif