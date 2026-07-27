#ifndef PTXSIM_CORE_WARP_CONTEXT_DISPATCH_H
#define PTXSIM_CORE_WARP_CONTEXT_DISPATCH_H

class WarpContext;  // forward decl (WarpContext lives in global namespace)
class StatementContext;  // forward decl (StatementContext lives in global namespace)

// Forward declarations of helper functions to enable friend declarations in
// WarpContext (refactor-warp-context C-18 Phase 3 extraction).
namespace warp_dispatch {
    void execute_warp_instruction(WarpContext* w, StatementContext& stmt, int target_pc);
}  // namespace warp_dispatch
#endif