#include "ptxsim/instruction_handlers/special_handler.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/thread_context.h"
#include <iostream>

void PRAGMA::execute(ThreadContext *context, StatementContext &stmt) {
    // PRAGMA指令暂未实现
    // 根据原始实现，这里不做任何操作
    auto ss = (StatementContext::PRAGMA *)stmt.statement;
    // 未实现的指令，暂时留空
}

void AT::execute(ThreadContext *context, StatementContext &stmt) {
    // AT指令暂未实现
    // 根据原始实现，这里不做任何操作
    auto ss = (StatementContext::AT *)stmt.statement;
    // 未实现的指令，暂时留空
}

void ATOM::execute(ThreadContext *context, StatementContext &stmt) {
    // ATOM指令暂未实现
    // 根据原始实现，这里不做任何操作
    auto ss = (StatementContext::ATOM *)stmt.statement;
    // 未实现的指令，暂时留空
}