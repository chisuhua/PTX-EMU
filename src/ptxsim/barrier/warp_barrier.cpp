// warp_barrier.cpp
#include "ptxsim/barrier/warp_barrier.h"
#include "utils/logger.h"
#include <algorithm>
#include <cstring>

namespace ptxsim {

WarpBarrier::WarpBarrier() {
    reset();
}

void WarpBarrier::init(uint32_t participation_mask, int reconvergence_pc, uint32_t barrier_pc) {
    participation_mask_ = participation_mask;
    arrived_mask_ = 0;
    expected_count_ = __builtin_popcount(participation_mask);
    arrived_count_ = 0;
    reconvergence_pc_ = reconvergence_pc;
    barrier_pc_ = barrier_pc;
    is_initialized_ = true;
    state_ = State::Initializing;

    PTX_DEBUG_EMU("WarpBarrier::init mask=0x%X reconv=%d barrier_pc=%u expected=%d",
                   participation_mask, reconvergence_pc, barrier_pc, expected_count_);
}

void WarpBarrier::arrive(int lane_id) {
    if (!is_initialized_) {
        PTX_ERROR_EMU("WarpBarrier::arrive called on uninitialized barrier");
        return;
    }

    if (lane_id < 0 || lane_id >= WARP_SIZE) {
        PTX_ERROR_EMU("WarpBarrier::arrive invalid lane_id=%d", lane_id);
        return;
    }

    uint32_t lane_mask = (1u << lane_id);

    if (arrived_mask_ & lane_mask) {
        PTX_DEBUG_EMU("WarpBarrier::arrive lane %d already arrived, skipping", lane_id);
        return;
    }

    arrived_mask_ |= lane_mask;
    arrived_count_++;

    PTX_DEBUG_EMU("WarpBarrier::arrive lane=%d arrived=%d/%d mask=0x%X",
                   lane_id, arrived_count_, expected_count_, arrived_mask_);

    if (state_ == State::Initializing || state_ == State::Waiting) {
        if (arrived_count_ < expected_count_) {
            state_ = State::Waiting;
        } else {
            state_ = State::Complete;
            PTX_INFO_EMU("WarpBarrier::complete mask=0x%X arrived=0x%X",
                         participation_mask_, arrived_mask_);
        }
    }
}

bool WarpBarrier::is_complete() const {
    if (!is_initialized_) return false;
    if (state_ == State::Complete || state_ == State::Released) return true;
    return (arrived_mask_ & participation_mask_) == participation_mask_;
}

bool WarpBarrier::needs_to_wait(int lane_id) const {
    if (!is_initialized_) return false;
    if (state_ == State::Complete || state_ == State::Released) return false;
    if (arrived_mask_ & (1u << lane_id)) return false;
    return true;
}

bool WarpBarrier::needs_to_wait() const {
    if (!is_initialized_) return false;
    if (state_ == State::Complete || state_ == State::Released) return false;
    return arrived_count_ < expected_count_;
}

void WarpBarrier::reset() {
    state_ = State::Uninitialized;
    participation_mask_ = 0;
    arrived_mask_ = 0;
    expected_count_ = 0;
    arrived_count_ = 0;
    reconvergence_pc_ = -1;
    barrier_pc_ = 0;
    is_initialized_ = false;
}

#ifdef PTX_DEBUG
std::string WarpBarrier::state_to_string() const {
    switch (state_) {
        case State::Uninitialized: return "Uninitialized";
        case State::Initializing: return "Initializing";
        case State::Waiting: return "Waiting";
        case State::Complete: return "Complete";
        case State::Released: return "Released";
        default: return "Unknown";
    }
}

void WarpBarrier::dump() const {
    PTX_DEBUG_EMU("WarpBarrier state=%s mask=0x%X arrived=0x%X expected=%d arrived=%d reconv=%d barrier_pc=%u",
                  state_to_string().c_str(),
                  participation_mask_,
                  arrived_mask_,
                  expected_count_,
                  arrived_count_,
                  reconvergence_pc_,
                  barrier_pc_);
}
#endif

} // namespace ptxsim