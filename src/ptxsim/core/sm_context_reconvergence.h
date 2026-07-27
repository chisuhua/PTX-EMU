#ifndef PTXSIM_CORE_SM_CONTEXT_RECONVERGENCE_H
#define PTXSIM_CORE_SM_CONTEXT_RECONVERGENCE_H

class WarpContext;  // forward decl (WarpContext lives in global namespace)

// Forward declaration of helper functions to enable friend declarations
// (god-class-refactor-sm-context C-2 Phase 1 dedup).
namespace sm_reconvergence {
    void drain_simt_and_update_active(WarpContext* warp);
}  // namespace sm_reconvergence
#endif