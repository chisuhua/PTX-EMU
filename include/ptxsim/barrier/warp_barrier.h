// warp_barrier.h
#ifndef WARP_BARRIER_H
#define WARP_BARRIER_H

#include "barrier_types.h"
#include <cstdint>
#include <string>

namespace ptxsim {

class WarpBarrier {
public:
    enum class State {
        Uninitialized,
        Initializing,
        Waiting,
        Complete,
        Released
    };

    WarpBarrier();

    void init(uint32_t participation_mask, int reconvergence_pc, uint32_t barrier_pc);

    void arrive(int lane_id);

    bool is_complete() const;

    State get_state() const { return state_; }

    int get_expected_count() const { return expected_count_; }
    int get_arrived_count() const { return arrived_count_; }
    uint32_t get_participation_mask() const { return participation_mask_; }
    uint32_t get_arrived_mask() const { return arrived_mask_; }
    int get_reconvergence_pc() const { return reconvergence_pc_; }
    uint32_t get_barrier_pc() const { return barrier_pc_; }

    bool needs_to_wait(int lane_id) const;

    bool needs_to_wait() const;

    void reset();

    uint32_t get_missing_mask() const {
        return participation_mask_ & ~arrived_mask_;
    }

    bool all_participants_arrived() const {
        if (!is_initialized_ || expected_count_ == 0) return false;
        return (arrived_mask_ & participation_mask_) == participation_mask_;
    }

    bool is_initialized() const { return is_initialized_; }

#ifdef PTX_DEBUG
    std::string state_to_string() const;
    void dump() const;
#endif

private:
    State state_;
    uint32_t participation_mask_;
    uint32_t arrived_mask_;
    int expected_count_;
    int arrived_count_;
    int reconvergence_pc_;
    uint32_t barrier_pc_;
    bool is_initialized_;
};

} // namespace ptxsim

#endif // WARP_BARRIER_H