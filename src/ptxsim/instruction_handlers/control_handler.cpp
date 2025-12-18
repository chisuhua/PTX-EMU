#include "ptxsim/instruction_handlers/control_handler.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/thread_context.h"
#include <cassert>
#include <iostream>

#ifdef DEBUGINTE
extern bool sync_thread;
#endif
#ifdef LOGINTE
extern bool IFLOG();
#endif

void BRA::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::BRA *)stmt.statement;

    auto iter = context->label2pc.find(ss->braTarget);
    assert(iter != context->label2pc.end());
    context->pc = iter->second -
                  1; // -1 because pc will be incremented after this instruction
}

void RET::execute(ThreadContext *context, StatementContext &stmt) {
    // 直接设置线程状态为退出
    context->state = EXIT;
}

void BAR::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::BAR *)stmt.statement;

    if (ss->barType == "sync") {
        context->state = BAR_SYNC;
#ifdef DEBUGINTE
        sync_thread = 1;
#endif
#ifdef LOGINTE
        if (IFLOG()) {
            std::cout << "INTE: Thread(" << context->ThreadIdx.x << ","
                      << context->ThreadIdx.y << "," << context->ThreadIdx.z
                      << ") in Block(" << context->BlockIdx.x << ","
                      << context->BlockIdx.y << "," << context->BlockIdx.z
                      << ") bar.sync" << std::endl;
        }
#endif
    } else {
        assert(false && "Unsupported barrier type");
    }
}