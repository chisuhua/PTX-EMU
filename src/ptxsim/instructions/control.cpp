#include "ptxsim/instruction_handlers.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include "ptxsim/warp_context.h"
#include <cmath>
#include <string>

// BRA is a BRANCH handler
void BraHandler::executeBranch(ThreadContext *context, const BranchInstr &instr) {
    // Check predicate condition if present
    if (!instr.predicate.empty()) {
        // Extract predicate register name (remove % prefix if present)
        std::string pred_name = instr.predicate;
        if (!pred_name.empty() && pred_name[0] == '%') {
            pred_name = pred_name.substr(1);
        }

        // Get predicate value from register bank
        std::string full_pred_name = pred_name;
        void *pred_data = context->register_bank_manager_->get_register(
            full_pred_name, context->warp_id_, context->lane_id_);

        if (pred_data == nullptr) {
            context->trace_status(ptxsim::log_level::error, "branch",
                                  "BRA: predicate register '%s' not found",
                                  instr.predicate.c_str());
            context->next_pc = context->pc + 1;
            return;
        }

        bool pred_value = (*(uint8_t *)pred_data) != 0;

        // Apply negation if needed (@!p means "when predicate is false")
        bool should_branch = instr.predicate_negated ? !pred_value : pred_value;

        if (!should_branch) {
            // Predicate false: do not branch, continue to next instruction
            context->next_pc = context->pc + 1;
            context->trace_status(ptxsim::log_level::debug, "branch",
                                  "BRA: predicate '%s' is false, not branching",
                                  instr.predicate.c_str());
            return;
        }

        // Predicate true: proceed with branch
        context->trace_status(ptxsim::log_level::debug, "branch",
                              "BRA: predicate '%s' is true, branching",
                              instr.predicate.c_str());
    }

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

// AT is now a BRANCH handler (conditional execution based on predicate)
// void AtHandler::executeBranch(ThreadContext *context, const BranchInstr &instr) {
//     // AT instruction: predicate prefix that determines whether to execute the next instruction
//     // In PTX, @p or @!p modifies whether the next instruction executes
//     
//     // TODO: Need to parse the actual predicate value from the instruction
//     // For now, we'll assume predicate is true (execute next instruction)
//     bool predicate_value = true;
//     
//     // Check if the predicate is negated (e.g., @!p)
//     // This information should be in the instruction qualifiers or operands
//     // For now, we'll assume it's not negated
//     
//     if (predicate_value) {
//         // Predicate is true: execute the next instruction
//         context->next_pc = context->pc + 1;
//         context->trace_status(ptxsim::log_level::debug, "predicate",
//                               "AT: predicate=true, executing next instruction at PC=%d", 
//                               context->next_pc);
//     } else {
//         // Predicate is false: skip the next instruction
//         context->next_pc = context->pc + 2;
//         context->trace_status(ptxsim::log_level::debug, "predicate",
//                               "AT: predicate=false, skipping to PC=%d", 
//                               context->next_pc);
//     }
// }
