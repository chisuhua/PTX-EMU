#ifndef PTXSIM_CONTEXTS_REGISTER_PREDICATE_H
#define PTXSIM_CONTEXTS_REGISTER_PREDICATE_H

#include "register/condition_code_register.h"
#include "register/register_bank_manager.h"
#include <memory>
#include <string>
#include <vector>

namespace ptxsim {
namespace contexts {

/**
 * @brief Register & predicate state POD: per-thread register bank access
 *        plus predicate / operand-collection scratch space.
 *
 * @details Groups fields involved in per-thread register access and
 *          operand-collection. The register bank manager is shared with
 *          sibling threads in the same warp/CTA. The condition-code
 *          register (cc_reg) tracks predicate results from setp/etc.
 *          operand_collected / operand_is_immediate_ / vecOp_phy_addrs are
 *          the per-thread scratch space used by collect_operands() and
 *          released by releaseAllOperands().
 *
 * @author PTX-EMU Team (T2-3 god-class split)
 * @date 2026-06-24
 */
struct RegisterPredicatePod {
    // Register bank manager (shared with sibling threads)
    std::shared_ptr<RegisterBankManager> register_bank_manager_;

    // Condition-code register (predicate state)
    ConditionCodeRegister cc_reg;

    // Operand collection scratch space (per-thread, lifetime = 1 instruction)
    std::vector<void *> operand_collected;
    std::vector<char> operand_is_immediate_;
    std::vector<std::vector<void *>> vecOp_phy_addrs;

    // Phase 0.3a (HSK-8 ack 738b412c): ThreadContext-local index-keyed cache
    // for operand physical addresses, parallel to operand_collected above.
    // Currently UNUSED — populated in Phase 0.3b, read in Phase 0.3c,
    // operand_phy_addr field removed in Phase 0.3d.
    std::vector<void *> operand_phy_cache_;

    // Cached destination register name (set during collect_operands,
    // consumed by setp handler to write the predicate)
    std::string dst_operand_reg_name_;
};

}  // namespace contexts
}  // namespace ptxsim

#endif  // PTXSIM_CONTEXTS_REGISTER_PREDICATE_H