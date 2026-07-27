#ifndef PTXSIM_CORE_WARP_CONTEXT_SIMT_H
#define PTXSIM_CORE_WARP_CONTEXT_SIMT_H

class WarpContext;  // forward decl (WarpContext lives in global namespace)

// Forward declarations of helper functions to enable friend declarations in
// WarpContext (refactor-warp-context C-18 Phase 2 extraction).
namespace warp_simt {
    bool check_reconvergence(WarpContext* w);
}  // namespace warp_simt
#endif