#ifndef SIMT_PC_MANAGER_H
#define SIMT_PC_MANAGER_H

#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/warp_state.h"
#include <vector>

class WarpContext;

// Manages per-thread PC and execution state, extracted from ThreadContext
// (Phase 1 of god-class-refactor-thread-context).
//
// All PC access delegates to warp_state.threads[lane_id] via WarpContext*,
// maintaining the DUAL STATE MECHANISM where warp_state is the SINGLE
// AUTHORITATIVE source for PC.
//
// Construction constraint (MR-5): must be constructed AFTER lane_id_ and
// warp_id_ are computed in ThreadContext::init().
class SimtPcManager {
public:
    // Constructor-injected dependencies (all non-owning pointers).
    // warp_ctx: for reading/writing warp_state.threads[lane_id]
    // lane_id: thread index within the warp
    // stmts: statement vector for is_valid_pc() / get_current_statement()
    SimtPcManager(WarpContext *warp_ctx, int lane_id,
                  std::vector<ptxemu::ir::StatementContext> *stmts)
        : warp_context_(warp_ctx), lane_id_(lane_id), state_(RUN),
          statements_(stmts) {}

    // ── PC accessors (delegate to warp_state) ─────────────────────────
    int get_pc() const;
    void set_pc(int new_pc);
    int get_next_pc() const;
    void set_next_pc(int new_next_pc);

    // Commit: pc ← next_pc. The sole method for normal PC advancement.
    void commit_pc();

    // ── Execution state ──────────────────────────────────────────────
    EXE_STATE get_state() const { return state_; }
    void set_state(EXE_STATE s) { state_ = s; }
    bool is_active() const { return state_ != EXIT; }
    bool is_exited() const { return state_ == EXIT; }
    bool is_at_barrier() const { return state_ == BAR_SYNC; }

    // ── Bidirectional warp_state sync ─────────────────────────────────
    // Read ThreadStatus from warp_state and translate to EXE_STATE.
    void sync_from_warp_state();
    // Write EXE_STATE to warp_state (with already_blocked guard).
    void sync_to_warp_state();

    // ── PC validation and statement query ────────────────────────────
    bool is_valid_pc() const;
    bool is_valid_pc(int p) const;
    int statements_size() const;
    ptxemu::ir::StatementContext *get_statement_at(int p);
    ptxemu::ir::StatementContext *get_current_statement();

    // ── Mutable accessors (MR-4: for set_warp_context fanout) ────────
    void set_warp_context(WarpContext *warp_ctx) { warp_context_ = warp_ctx; }
    WarpContext *get_warp_context() const { return warp_context_; }

private:
    WarpContext *warp_context_;
    int lane_id_;
    EXE_STATE state_;
    std::vector<ptxemu::ir::StatementContext> *statements_;
};

#endif // SIMT_PC_MANAGER_H