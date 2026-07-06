#ifndef REGISTER_ACCESS_LAYER_H
#define REGISTER_ACCESS_LAYER_H

#include "register/register_bank_manager.h"
#include <memory>
#include <vector>

class Dim3;
struct RegOperand;
enum class Qualifier : int;

// Encapsulates register lookup and allocation, extracted from ThreadContext
// (Phase 2 of god-class-refactor-thread-context).
//
// Handlers access cc_reg directly on ThreadContext (138 references), so
// cc_reg remains on ThreadContext for now (to be addressed in Phase 3).
class RegisterAccessLayer {
public:
    // Constructor-injected dependencies.
    // bank_mgr: shared register bank (may be nullptr initially, set later)
    // warp_id, lane_id: thread identity for per-thread register lookup
    // tIdx, bIdx, gDim, bDim: thread/block identifiers for special register
    //   resolution (tid.x, ctaid.x, etc.)
    RegisterAccessLayer(std::shared_ptr<RegisterBankManager> bank_mgr,
                        int warp_id, int lane_id,
                        const Dim3 &tIdx, const Dim3 &bIdx,
                        const Dim3 &gDim, const Dim3 &bDim);

    // Resolve a register operand to its backing memory address.
    // Handles special registers (tid.x, ctaid.x, ntid.x, nctaid.x, etc.)
    // before falling through to RegisterBankManager.
    void *acquire_register(const RegOperand &reg,
                           std::vector<Qualifier> qualifier);

    // ── Register bank access ──────────────────────────────────────
    void set_register_bank_manager(std::shared_ptr<RegisterBankManager> mgr) {
        register_bank_manager_ = std::move(mgr);
    }
    std::shared_ptr<RegisterBankManager> get_register_bank_manager() const {
        return register_bank_manager_;
    }

private:
    std::shared_ptr<RegisterBankManager> register_bank_manager_;
    int warp_id_;
    int lane_id_;

    // Non-owning pointers/references for special register resolution
    const Dim3 &thread_idx_;
    const Dim3 &block_idx_;
    const Dim3 &grid_dim_;
    const Dim3 &block_dim_;
};

#endif // REGISTER_ACCESS_LAYER_H