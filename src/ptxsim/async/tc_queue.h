// src/ptxsim/async/tc_queue.h
// Phase 0.4 (Fix #8): Blackwell per-CTA async tensor core command queue.
//
// Maintains a per-CTA commit-group counter (std::atomic<uint64_t>) and
// a pending-waiter list for tcgen05.* async commit/wait primitives.
//
// Design:
//   - std::atomic<uint64_t> commit_group_counter_ with CAS fetch_max
//     (monotonic, no mutex on counter per Oracle Q2 hypothesis 1).
//   - std::mutex ONLY for pending_waiters_ list (avoids nested lock
//     per ptx-lessons-learned §2 — counter and list use separate
//     synchronization primitives).
//   - wait() stores (lane_id, group_id, completion_pc=current_pc+1,
//     warp_ctx) in pending_waiters_ and sets warp_state is_blocked +
//     status=Blocked per-lane. Follows BarWarpSyncHandler pattern
//     (is_blocked + status only; NO set_state(BAR_SYNC) — TcQueue is
//     not a CTA-level barrier and does not need BAR_SYNC fallback
//     path per Oracle Q1 hypothesis 2).
//   - commit() mirrors release_warp_barrier post-unblock per-lane:
//     advance_thread_pc to pre-captured completion_pc, clear is_blocked,
//     set status=Active, set is_active=true.
//     NO set_active_mask (AGENTS.md: OR semantics owner is
//     BarrierModule::release_warp_barrier).
//   - completion_pc is captured at wait() call time (NOT at release
//     time) to avoid get_pc() drift per ptx-barrier-mechanism "PC
//     管理:多层覆写链" and Oracle Q4 hypothesis 2.
//   - NO new exec state (per Decision 7; BAR_SYNC reuse pattern only).
//
// Per-CTA isolation: each CTA gets its own TcQueue instance; operations
// within one CTA's queue do not affect other CTAs.
//
// Consumed by tcgen05.* in Phase 1-3.

#ifndef PTXSIM_ASYNC_TC_QUEUE_H
#define PTXSIM_ASYNC_TC_QUEUE_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

class WarpContext;

class TcQueue {
public:
    using group_id_t = uint64_t;
    using lane_id_t = uint32_t;

    TcQueue();
    ~TcQueue();

    void commit(group_id_t group_id);

    void wait(WarpContext* warp_ctx, lane_id_t lane_id, group_id_t group_id);

    void clear();

    group_id_t current_counter() const;
    size_t pending_count() const;

private:
    struct PendingWaiter {
        lane_id_t lane_id;
        group_id_t waited_group_id;
        int completion_pc;
        WarpContext* warp_ctx;
    };

    std::atomic<group_id_t> commit_group_counter_;
    mutable std::mutex mu_;
    std::vector<PendingWaiter> pending_waiters_;
};

#endif  // PTXSIM_ASYNC_TC_QUEUE_H