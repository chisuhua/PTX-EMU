#include "ptxsim/instruction_handlers.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include "ptxsim/warp_context.h"
#include <cmath>
#include <string>

void BraHandler::executeBranch(ThreadContext *context, const BranchInstr &instr) {
    uint32_t taken_mask = 0;
    uint32_t not_taken_mask = 0;
    
    for (int i = 0; i < 32; i++) {
        bool should_branch = true;
        
        if (!instr.predicate.empty()) {
            std::string pred_name = instr.predicate;
            if (!pred_name.empty() && pred_name[0] == '%') {
                pred_name = pred_name.substr(1);
            }
            
            std::string full_pred_name = pred_name;
            void *pred_data = context->register_bank_manager_->get_register(
                full_pred_name, context->warp_id_, i);
            
            if (pred_data != nullptr) {
                bool pred_value = (*(uint8_t *)pred_data) != 0;
                should_branch = instr.predicate_negated ? !pred_value : pred_value;
            } else {
                should_branch = !instr.predicate_negated;
            }
        }
        
        if (should_branch) {
            taken_mask |= (1u << i);
        } else {
            not_taken_mask |= (1u << i);
        }
    }
    
    bool is_divergent = (taken_mask != 0) && (not_taken_mask != 0);
    
    if (is_divergent) {
        SIMTStackEntry entry;
        entry.branch_pc = context->pc;
        entry.reconvergence_pc = instr.reconvergence_pc;
        entry.active_mask = taken_mask;
        entry.return_mask = context->warp_context_->get_exec_mask();
        entry.return_pc = instr.reconvergence_pc;
        
        context->warp_context_->get_simt_stack().push(entry);
        
        int target_pc = context->label2pc.count(instr.target) > 0 
                       ? context->label2pc.at(instr.target) 
                       : context->pc + 1;
        
        for (int i = 0; i < 32; i++) {
            if (taken_mask & (1u << i)) {
                context->warp_context_->set_thread_pc(i, target_pc);
            } else if (not_taken_mask & (1u << i)) {
                context->warp_context_->set_thread_pc(i, context->pc + 1);
            }
        }
        
        context->warp_context_->set_exec_mask(taken_mask);
    } else {
        int target_pc = (taken_mask != 0) 
                       ? (context->label2pc.count(instr.target) > 0 
                          ? context->label2pc.at(instr.target) 
                          : context->pc + 1)
                       : context->pc + 1;
        
        for (int i = 0; i < 32; i++) {
            if (context->warp_context_->is_lane_active(i)) {
                context->warp_context_->set_thread_pc(i, target_pc);
            }
        }
    }
}
