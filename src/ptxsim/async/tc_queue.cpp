// src/ptxsim/async/tc_queue.cpp
// Phase 0.4 (Fix #8): per-CTA async tensor core command queue implementation.
//
// Maintains a per-CTA commit-group counter and pending-waiter list for
// tcgen05.* async commit/wait primitives. Counter uses std::atomic with
// CAS fetch_max (monotonic). Pending list uses std::mutex (avoids nested
// lock per ptx-lessons-learned §2).
//
// wait() pattern (BarWarpSyncHandler-style):
//   - captures pc+1 at call time (NOT at release time — avoids PC drift)
//   - stores waiter in pending_waiters_ (with warp_ctx pointer for release)
//   - sets is_blocked=true, status=Blocked per-lane
//   - NO set_state(BAR_SYNC) — TcQueue is not a CTA-level barrier and does
//     not need the BAR_SYNC→is_blocked fallback path per Oracle Q1 hypothesis 2
//   - NO set_active_mask — OR semantics are owned by
//     BarrierModule::release_warp_barrier (AGENTS.md)
//
// commit() pattern (mirrors release_warp_barrier post-unblock per-lane):
//   - CAS atomic counter update (monotonic max)
//   - collects waiters whose waited_group_id ≤ new_counter
//   - releases each: advance_thread_pc to pre-captured completion_pc,
//     clear is_blocked, set status=Active, is_active=true
//   - NO OR-set_active_mask (TcQueue is per-CTA per-lane, not warp-wide)

#include "ptxsim/async/tc_queue.h"

#include <stdexcept>

#include "ptxsim/thread_state.h"
#include "ptxsim/warp_context.h"

inline void throw_error(const char* msg) { throw std::runtime_error(msg); }

TcQueue::TcQueue()
    : commit_group_counter_(0) {}

TcQueue::~TcQueue() = default;

void TcQueue::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    commit_group_counter_.store(0, std::memory_order_release);
    pending_waiters_.clear();
}

TcQueue::group_id_t TcQueue::current_counter() const {
    return commit_group_counter_.load(std::memory_order_acquire);
}

size_t TcQueue::pending_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return pending_waiters_.size();
}

void TcQueue::commit(group_id_t group_id) {
    group_id_t prev = commit_group_counter_.load(std::memory_order_acquire);
    while (prev < group_id) {
        if (commit_group_counter_.compare_exchange_weak(
                prev, group_id, std::memory_order_acq_rel)) {
            break;
        }
    }

    group_id_t new_counter = commit_group_counter_.load(
        std::memory_order_acquire);

    std::vector<PendingWaiter> to_wake;
    {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<PendingWaiter> remaining;
        for (const auto& w : pending_waiters_) {
            if (w.waited_group_id <= new_counter) {
                to_wake.push_back(w);
            } else {
                remaining.push_back(w);
            }
        }
        pending_waiters_ = std::move(remaining);
    }

    for (const auto& w : to_wake) {
        w.warp_ctx->advance_thread_pc(w.lane_id, w.completion_pc);
        auto& ts = w.warp_ctx->get_warp_state().threads[w.lane_id];
        ts.is_blocked = false;
        ts.status = ptxsim::ThreadStatus::Active;
        ts.is_active = true;
    }
}

void TcQueue::wait(WarpContext* warp_ctx, lane_id_t lane_id,
                   group_id_t group_id) {
    if (!warp_ctx) {
        throw_error("TcQueue::wait: null WarpContext pointer");
    }
    if (lane_id >= static_cast<lane_id_t>(WarpContext::WARP_SIZE)) {
        throw_error("TcQueue::wait: lane_id out of range");
    }

    int completion_pc = static_cast<int>(warp_ctx->get_thread_pc(lane_id)) + 1;

    // Oracle §29 fix: check counter BEFORE pushing to pending_waiters_.
    // If the commit_group_counter already satisfies the wait group_id,
    // return immediately — no need to block. Without this, commit(2)
    // followed by wait(1) would leave a stale waiter in pending_waiters_
    // (pending_count() == 1 even though the counter was sufficient).
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (commit_group_counter_.load(std::memory_order_acquire) >= group_id) {
            return;
        }
        pending_waiters_.push_back(
            {lane_id, group_id, completion_pc, warp_ctx});
    }

    auto& ts = warp_ctx->get_warp_state().threads[lane_id];
    ts.is_blocked = true;
    ts.status = ptxsim::ThreadStatus::Blocked;
}