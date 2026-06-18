// barrier_module.h
#ifndef BARRIER_MODULE_H
#define BARRIER_MODULE_H

#include "ptxsim/barrier/warp_barrier.h"
#include "ptxsim/barrier/cta_barrier.h"
#include "ptxsim/warp_context.h"
#include <array>
#include <memory>
#include <vector>

namespace ptxsim {

class ThreadContext;
class CTAContext;  // forward declare for release_cta_barrier signature

class BarrierModule {
public:
    static constexpr int MAX_WARP_BARRIERS = 4;
    static constexpr int MAX_CTA_BARRIERS = 16;

    BarrierModule();

    // Warp Barrier
    WarpBarrier* init_warp_barrier(int warp_barrier_id,
                                   uint32_t participation_mask,
                                   int reconvergence_pc,
                                   uint32_t barrier_pc);

    WarpBarrier* get_warp_barrier(int warp_barrier_id);

    bool arrive_at_warp_barrier(int warp_barrier_id, int lane_id);

    bool is_warp_barrier_complete(int warp_barrier_id) const;

    bool warp_barrier_needs_wait(int warp_barrier_id, int lane_id) const;

    void release_warp_barrier(int warp_barrier_id, WarpContext* warp_ctx);

    // CTA Barrier
    // release_cta_barrier advances per-thread PC for every arrived thread via
    // cta_ctx -> thread -> warp_context -> advance_thread_pc(lane, post_barrier_pc),
    // and sets ThreadContext::state = RUN so the scheduler can resume execution.
    // MUST only be called when is_cta_barrier_complete() returned true.
    void release_cta_barrier(int cta_barrier_id, CTAContext* cta_ctx,
                             int post_barrier_pc);

    CTABarrier* init_cta_barrier(int cta_barrier_id,
                                 int total_threads,
                                 int warp_count);

    CTABarrier* get_cta_barrier(int cta_barrier_id);

    bool arrive_at_cta_barrier(int cta_barrier_id, ThreadContext* thread);

    bool is_cta_barrier_complete(int cta_barrier_id) const;

    // 状态查询
    int get_active_warp_barrier_count() const;
    int get_active_cta_barrier_count() const;

    void reset_all();

#ifdef PTX_DEBUG
    void dump() const;
#endif

private:
    std::array<WarpBarrier, MAX_WARP_BARRIERS> warp_barriers_;
    std::vector<std::unique_ptr<CTABarrier>> cta_barriers_;
};

} // namespace ptxsim

#endif // BARRIER_MODULE_H