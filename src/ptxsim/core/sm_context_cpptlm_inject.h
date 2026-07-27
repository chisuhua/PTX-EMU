#ifndef PTXSIM_CORE_SM_CONTEXT_CPPTLM_INJECT_H
#define PTXSIM_CORE_SM_CONTEXT_CPPTLM_INJECT_H

#include "ptxsim/tensor_core_interface.h"  // brings TcPrecision declaration

class WarpContext;  // forward decl (WarpContext lives in global namespace)
class SMContext;
class IPipelineLatencyProvider;
class ITensorCoreTiming;
struct StatementContext;  // global namespace

// Forward declaration of helper functions (god-class-refactor-sm-context C-2
// Phase 2: ADR-0020 cpptlm injection extraction).
namespace sm_cpptlm_inject {

// Step B: compute blocked cycles for the warp based on pipeline/tc latency
// providers. No-op when both providers are nullptr (preserves lessons-learned
// §14 byte-identical fallback contract — 4-branch test in
// tests/unit/cpputlm/test_step_b_set_blocked_cycles.cpp must stay green).
void step_b_set_blocked_cycles(IPipelineLatencyProvider* pipeline,
                               ITensorCoreTiming* tc,
                               WarpContext* warp,
                               const StatementContext& stmt);

}  // namespace sm_cpptlm_inject
#endif