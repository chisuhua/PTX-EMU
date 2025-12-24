// InstructionHandlers_decl.h
#ifndef INSTRUCTION_HANDLE_H
#define INSTRUCTION_HANDLE_H
#include "../ptx_ir/statement_context.h"
#include <memory>
#include <vector>

class ThreadContext;

class InstructionHandler {
public:
    virtual ~InstructionHandler() = default;

    // 分阶段执行接口
    virtual bool prepare(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool execute(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool commit(ThreadContext *context, StatementContext &stmt) = 0;

    // 保持向后兼容的execute接口 - 使用不同名称避免重载冲突
    virtual void execute_full(ThreadContext *context, StatementContext &stmt) {
        if (!prepare(context, stmt)) {
            return;
        }
        if (!execute(context, stmt)) {
            return;
        }
        commit(context, stmt);
    }
    
    // 处理指令操作的通用函数 - 根据实际使用调整参数
    virtual void process_operation(ThreadContext *context, void **operands, const std::vector<Qualifier> &qualifiers) = 0;
};

#endif