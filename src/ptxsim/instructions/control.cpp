#include "ptxsim/instruction_handlers.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include "ptxsim/warp_context.h"
#include <cmath>
#include <string>

// BRA is a BRANCH handler
void BRA_Handler::executeBranch(ThreadContext *context, const BranchInstr &instr) {
    // Look up the label in the label-to-PC map
    auto it = context->label2pc.find(instr.target);
    if (it != context->label2pc.end()) {
        // Found the label, set next PC to the target address
        context->next_pc = it->second;
        
        // Debug output
        context->trace_status(ptxsim::log_level::debug, "branch", 
                              "BRA: jumping to label '%s' at PC=%d", 
                              instr.target.c_str(), it->second);
    } else {
        // Label not found - this is an error
        // In real PTX, this should not happen if the code was properly parsed
        // For now, fall back to PC + 1 (no jump)
        context->next_pc = context->pc + 1;
        
        // Error message
        context->trace_status(ptxsim::log_level::error, "branch",
                              "BRA: label '%s' not found, cannot jump",
                              instr.target.c_str());
    }
}

// AT is a PREDICATE_PREFIX handler (SimpleHandler)
void AT_Handler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    // AT instruction sets a predicate prefix for the next instruction
    // In PTX, @p or @!p modifies whether the next instruction executes
    
    // For now, this is a simplified implementation
    // Just move to the next instruction
    context->next_pc = context->pc + 1;
    
    // Debug output
    context->trace_status(ptxsim::log_level::debug, "predicate",
                          "AT: predicate prefix instruction");
}
