// barrier_module.cpp
#include "ptxsim/barrier/barrier_module.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "utils/logger.h"
#include <sstream>
#include <set>

namespace ptxsim {

BarrierModule::BarrierModule() {
    cta_barriers_.resize(MAX_CTA_BARRIERS);
    for (int i = 0; i < MAX_CTA_BARRIERS; ++i) {
        cta_barriers_[i] = std::make_unique<CTABarrier>(i);
    }
    PTX_DEBUG_EMU("BarrierModule::BarrierModule created with %d warp barriers, %d CTA barriers",
                  MAX_WARP_BARRIERS, MAX_CTA_BARRIERS);
}

WarpBarrier* BarrierModule::init_warp_barrier(int warp_barrier_id,
                                              uint32_t participation_mask,
                                              int reconvergence_pc,
                                              uint32_t barrier_pc) {
    if (warp_barrier_id < 0 || warp_barrier_id >= MAX_WARP_BARRIERS) {
        PTX_ERROR_EMU("BarrierModule::init_warp_barrier invalid id=%d", warp_barrier_id);
        return nullptr;
    }

    WarpBarrier* wbar = &warp_barriers_[warp_barrier_id];
    wbar->init(participation_mask, reconvergence_pc, barrier_pc);

    PTX_DEBUG_EMU("BarrierModule::init_warp_barrier id=%d mask=0x%X reconv_pc=%d barrier_pc=%u "
                  "expected_threads=%d",
                  warp_barrier_id, participation_mask, reconvergence_pc, barrier_pc,
                  wbar->get_expected_count());

    return wbar;
}

WarpBarrier* BarrierModule::get_warp_barrier(int warp_barrier_id) {
    if (warp_barrier_id < 0 || warp_barrier_id >= MAX_WARP_BARRIERS) {
        return nullptr;
    }
    return &warp_barriers_[warp_barrier_id];
}

bool BarrierModule::arrive_at_warp_barrier(int warp_barrier_id, int lane_id) {
    WarpBarrier* wbar = get_warp_barrier(warp_barrier_id);
    if (!wbar) {
        PTX_ERROR_EMU("BarrierModule::arrive_at_warp_barrier id=%d lane=%d - wbar is null",
                      warp_barrier_id, lane_id);
        return false;
    }

    int arrived_before = wbar->get_arrived_count();
    wbar->arrive(lane_id);
    int arrived_after = wbar->get_arrived_count();
    bool complete = wbar->is_complete();

    PTX_DEBUG_EMU("BarrierModule::arrive_at_warp_barrier id=%d lane=%d "
                  "arrived=%d/%d complete=%s missing_mask=0x%X",
                  warp_barrier_id, lane_id,
                  arrived_after, wbar->get_expected_count(),
                  complete ? "YES" : "NO",
                  wbar->get_missing_mask());

    return complete;
}

bool BarrierModule::is_warp_barrier_complete(int warp_barrier_id) const {
    if (warp_barrier_id < 0 || warp_barrier_id >= MAX_WARP_BARRIERS) {
        return false;
    }
    return warp_barriers_[warp_barrier_id].is_complete();
}

bool BarrierModule::warp_barrier_needs_wait(int warp_barrier_id, int lane_id) const {
    if (warp_barrier_id < 0 || warp_barrier_id >= MAX_WARP_BARRIERS) {
        return false;
    }
    return warp_barriers_[warp_barrier_id].needs_to_wait(lane_id);
}

void BarrierModule::release_warp_barrier(int warp_barrier_id, WarpContext* warp_ctx) {
    WarpBarrier* wbar = get_warp_barrier(warp_barrier_id);
    if (!wbar || !warp_ctx) {
        PTX_ERROR_EMU("BarrierModule::release_warp_barrier id=%d - wbar=%p warp_ctx=%p",
                      warp_barrier_id, (void*)wbar, (void*)warp_ctx);
        return;
    }

    if (!wbar->is_complete()) {
        PTX_ERROR_EMU("BarrierModule::release_warp_barrier called on incomplete barrier id=%d",
                      warp_barrier_id);
        return;
    }

    int reconv_pc = wbar->get_reconvergence_pc();
    uint32_t arrived_mask = wbar->get_arrived_mask();
    int arrived_count = wbar->get_arrived_count();

    PTX_INFO_EMU("BarrierModule::release_warp_barrier id=%d COMPLETE - releasing %d threads to PC=%d "
                 "arrived_mask=0x%X",
                 warp_barrier_id, arrived_count, reconv_pc, arrived_mask);

    // BUG-POSTBARRIER-TWOHALVES: OR with existing active_mask (preserves
    // lanes already released by a prior barrier call when a divergent warp
    // hits the same barrier in two halves). MUST live in caller — ret
    // handler relies on set_active_mask overwrite semantics (0u to clear).
    warp_ctx->set_active_mask(
        warp_ctx->get_active_mask() | arrived_mask);
    warp_ctx->set_exec_mask(arrived_mask);

    for (int i = 0; i < WarpContext::WARP_SIZE; ++i) {
        if (arrived_mask & (1u << i)) {
            warp_ctx->advance_thread_pc(i, reconv_pc);

            // Unblock and reactivate lanes so the scheduler recognizes them
            // as executable after barrier release. This matches the old
            // paths A/B (barrier.cpp:189-190, 252-253) and is symmetric
            // with release_cta_barrier (barrier_module.cpp:194-196).
            // Without these, lanes remain in blocked/Blocked state and the
            // scheduler permanently skips them (BUG-RECONVERGENCE-SIMPLEGEMM
            // failure mode, lessons-learned.md §1: cross-module state
            // translation).
            auto& ts = warp_ctx->get_warp_state().threads[i];
            ts.is_blocked = false;
            ts.status = ptxsim::ThreadStatus::Active;
            ts.is_active = true;  // required: get_lanes_by_pc() filters on is_active

            PTX_DEBUG_EMU("  released lane %d -> PC=%d (unblocked, active)", i, reconv_pc);
        }
    }

    wbar->reset();

    PTX_DEBUG_EMU("BarrierModule::release_warp_barrier id=%d RESET done", warp_barrier_id);
}

CTABarrier* BarrierModule::init_cta_barrier(int cta_barrier_id,
                                            int total_threads,
                                            int warp_count) {
    if (cta_barrier_id < 0 || cta_barrier_id >= MAX_CTA_BARRIERS) {
        PTX_ERROR_EMU("BarrierModule::init_cta_barrier invalid id=%d", cta_barrier_id);
        return nullptr;
    }

    CTABarrier* ctabar = cta_barriers_[cta_barrier_id].get();
    ctabar->init(cta_barrier_id, total_threads, warp_count);
    return ctabar;
}

CTABarrier* BarrierModule::get_cta_barrier(int cta_barrier_id) {
    if (cta_barrier_id < 0 || cta_barrier_id >= MAX_CTA_BARRIERS) {
        return nullptr;
    }
    return cta_barriers_[cta_barrier_id].get();
}

bool BarrierModule::arrive_at_cta_barrier(int cta_barrier_id, ThreadContext* thread) {
    CTABarrier* ctabar = get_cta_barrier(cta_barrier_id);
    if (!ctabar) {
        PTX_ERROR_EMU("arrive_at_cta_barrier id=%d NULL ctabar", cta_barrier_id);
        return false;
    }

    bool complete = ctabar->arrive(thread);
    PTX_DEBUG_EMU("arrive_at_cta_barrier id=%d thread=%p lane=%d complete=%d",
                  cta_barrier_id, (void*)thread, thread ? thread->lane_id_ : -1, (int)complete);
    return complete;
}

bool BarrierModule::is_cta_barrier_complete(int cta_barrier_id) const {
    if (cta_barrier_id < 0 || cta_barrier_id >= MAX_CTA_BARRIERS) {
        return false;
    }
    return cta_barriers_[cta_barrier_id]->is_complete();
}

void BarrierModule::release_cta_barrier(int cta_barrier_id,
                                        CTAContext* cta_ctx,
                                        int post_barrier_pc) {
    CTABarrier* ctabar = get_cta_barrier(cta_barrier_id);
    if (!ctabar) {
        PTX_ERROR_EMU("BarrierModule::release_cta_barrier id=%d - ctabar is null",
                      cta_barrier_id);
        return;
    }

    if (!ctabar->is_complete()) {
        PTX_ERROR_EMU("BarrierModule::release_cta_barrier called on incomplete barrier id=%d",
                      cta_barrier_id);
        return;
    }

    // Snapshot the waiting set BEFORE releasing — get_waiting_threads() returns
    // a const reference, but reset() will clear the set, so we must copy first.
    std::set<ThreadContext*> arrived = ctabar->get_waiting_threads();
    int released_count = 0;

    for (ThreadContext* thread : arrived) {
        if (!thread) continue;
        thread->set_state(RUN);
        if (thread->warp_context_ != nullptr) {
            int lane = thread->lane_id_;
            auto& ts = thread->warp_context_->get_warp_state().threads[lane];
            ts.is_blocked = false;
            ts.status = ptxsim::ThreadStatus::Active;
            ts.is_active = true;  // required: get_lanes_by_pc() filters on is_active
            thread->warp_context_->advance_thread_pc(lane, post_barrier_pc);
            // Sync thread_context.next_pc with the warp_state advance.
            // Without this, sync_to_warp_state() would overwrite
            // warp_state.next_pc with the stale thread_context value
            // (the barrier's blocking PC), breaking is_warp_ready_to_fetch()
            // and causing the scheduler to permanently skip this warp.
            thread->set_next_pc(post_barrier_pc);
            released_count++;
        } else {
            PTX_ERROR_EMU("release_cta_barrier thread=%p has no warp_context_", (void*)thread);
        }
    }

    // Call update_active_mask on each affected warp so get_lanes_by_pc()
    // sees the released threads as schedulable on their next tick.
    std::set<WarpContext*> updated_warps;
    for (ThreadContext* thread : arrived) {
        if (thread && thread->warp_context_) {
            if (updated_warps.insert(thread->warp_context_).second) {
                thread->warp_context_->update_active_mask();
            }
        }
    }

    ctabar->reset();
}

int BarrierModule::get_active_warp_barrier_count() const {
    int count = 0;
    for (int i = 0; i < MAX_WARP_BARRIERS; ++i) {
        if (warp_barriers_[i].is_initialized()) {
            count++;
        }
    }
    return count;
}

int BarrierModule::get_active_cta_barrier_count() const {
    int count = 0;
    for (int i = 0; i < MAX_CTA_BARRIERS; ++i) {
        if (cta_barriers_[i]->get_arrived_count() > 0) {
            count++;
        }
    }
    return count;
}

void BarrierModule::reset_all() {
    for (auto& wbar : warp_barriers_) {
        wbar.reset();
    }
    for (auto& ctabar : cta_barriers_) {
        ctabar->reset();
    }
}

#ifdef PTX_DEBUG
void BarrierModule::dump() const {
    PTX_DEBUG_EMU("=== BarrierModule ===");
    PTX_DEBUG_EMU("Active warp barriers: %d", get_active_warp_barrier_count());
    for (int i = 0; i < MAX_WARP_BARRIERS; ++i) {
        if (warp_barriers_[i].is_initialized()) {
            PTX_DEBUG_EMU("  warp_barrier[%d]:", i);
            warp_barriers_[i].dump();
        }
    }
    PTX_DEBUG_EMU("Active CTA barriers: %d", get_active_cta_barrier_count());
    for (int i = 0; i < MAX_CTA_BARRIERS; ++i) {
        if (cta_barriers_[i]->get_arrived_count() > 0) {
            PTX_DEBUG_EMU("  cta_barrier[%d]: %s", i, cta_barriers_[i]->dump().c_str());
        }
    }
}
#endif

} // namespace ptxsim