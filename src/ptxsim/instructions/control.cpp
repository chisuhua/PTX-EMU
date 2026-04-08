#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include <string>

void BraHandler::executeBranch(ThreadContext *context, const BranchInstr &instr) {
    WarpContext* warp_ctx = context->warp_context_;
    
    int target_pc = -1;
    auto it = context->label2pc.find(instr.target);
    if (it != context->label2pc.end()) {
        target_pc = it->second;
    } else {
        target_pc = context->pc + 1;
    }
    
    warp_ctx->handle_branch(
        instr.predicate,
        instr.predicate_negated,
        target_pc,
        instr.reconvergence_pc
    );
    
    context->pc = warp_ctx->get_thread_pc(context->lane_id_);
}
