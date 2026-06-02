#ifndef PTXSIM_TESTING_SCHEDULER_UTILS_H
#define PTXSIM_TESTING_SCHEDULER_UTILS_H

#include "ptxsim/warp_context.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/gpu_context.h"

#include <vector>
#include <map>
#include <memory>

namespace ptxsim::testing {

enum class StepResult { Continue, BarrierHit, Converged, Diverged, Complete };

// step_warp — 完全模拟 sm_context.cpp 调度器算法
// Returns the PC that was executed
inline int step_warp(WarpContext* w, std::vector<StatementContext>& v) {
    auto m = w->get_lanes_by_pc();
    // REQUIRE_FALSE(m.empty());
    int pick = m.begin()->first;
    auto& ws = w->get_warp_state();
    for (auto& [pc, lanes] : m) {
        bool ok = true;
        for (int l : lanes) { if (ws.threads[l].is_blocked) { ok = false; break; } }
        if (ok) { pick = pc; break; }
    }
    w->execute_warp_instruction(v[pick], pick);
    while (w->check_reconvergence()) {}
    return pick;
}

// make_kernel_request — 从 StatementContext 向量创建 KernelLaunchRequest
inline KernelLaunchRequest make_kernel_request(
    std::vector<StatementContext>& statements,
    std::map<std::string, Symtable*>& name2Sym,
    std::map<std::string, int>& label2pc,
    void** args = nullptr,
    Dim3 gridDim = {1, 1, 1},
    Dim3 blockDim = {32, 1, 1},
    size_t sharedMem = 0) {

  KernelLaunchRequest req;
  req.args = args;
  req.gridDim = gridDim;
  req.blockDim = blockDim;
  req.statements = &statements;
  req.name2Sym = std::make_shared<std::map<std::string, Symtable*>>(name2Sym);
  req.label2pc = std::make_shared<std::map<std::string, int>>(label2pc);
  req.shared_mem_size = sharedMem;
  return req;
}

// run_until_converged — 运行 warp 直到汇聚或达到最大步数
// Returns true if converged, false if max_steps reached
inline bool run_until_converged(WarpContext* warp,
                                const std::vector<StatementContext>& stmts,
                                int max_steps = 1000) {
    for (int i = 0; i < max_steps; ++i) {
        uint32_t mask = warp->get_active_mask();
        if (mask == 1) {
            return true;
        }
        step_warp(warp, const_cast<std::vector<StatementContext>&>(stmts));
    }
    return false;
}

} // namespace ptxsim::testing

#endif // PTXSIM_TESTING_SCHEDULER_UTILS_H
