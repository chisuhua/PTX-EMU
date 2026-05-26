#include "ptxsim/bsync_state.h"

namespace ptxsim {

void BsyncManager::bssy(uint32_t barrier_id, uint32_t thread_mask) {
    BsyncState& state = barriers_[barrier_id];
    state.barrier_id = barrier_id;
    state.waiting_threads_mask = 0;
    state.total_threads = __builtin_popcount(thread_mask);
    state.suspended_pc = 0;
    state.is_released = false;
}

bool BsyncManager::bsync(uint32_t barrier_id, uint32_t lane_id, uint32_t current_pc) {
    auto it = barriers_.find(barrier_id);
    if (it == barriers_.end()) {
        return false;
    }
    
    BsyncState& state = it->second;
    state.waiting_threads_mask |= (1u << lane_id);
    state.suspended_pc = current_pc;
    return true;
}

bool BsyncManager::check_release(uint32_t barrier_id) {
    auto it = barriers_.find(barrier_id);
    if (it == barriers_.end()) {
        return false;
    }
    return it->second.all_arrived();
}

void BsyncManager::release(uint32_t barrier_id) {
    auto it = barriers_.find(barrier_id);
    if (it != barriers_.end()) {
        it->second.is_released = true;
        it->second.waiting_threads_mask = 0;
    }
}

BsyncState* BsyncManager::get_state(uint32_t barrier_id) {
    auto it = barriers_.find(barrier_id);
    if (it != barriers_.end()) {
        return &it->second;
    }
    return nullptr;
}

const BsyncState* BsyncManager::get_state(uint32_t barrier_id) const {
    auto it = barriers_.find(barrier_id);
    if (it != barriers_.end()) {
        return &it->second;
    }
    return nullptr;
}

bool BsyncManager::is_waiting(uint32_t barrier_id, uint32_t lane_id) const {
    auto it = barriers_.find(barrier_id);
    if (it == barriers_.end()) {
        return false;
    }
    return (it->second.waiting_threads_mask & (1u << lane_id)) != 0;
}

uint32_t BsyncManager::get_waiting_mask(uint32_t barrier_id) const {
    auto it = barriers_.find(barrier_id);
    if (it == barriers_.end()) {
        return 0;
    }
    return it->second.waiting_threads_mask;
}

void BsyncManager::cleanup() {
    for (auto it = barriers_.begin(); it != barriers_.end(); ) {
        if (it->second.is_released) {
            it = barriers_.erase(it);
        } else {
            ++it;
        }
    }
}

void BsyncManager::reset() {
    barriers_.clear();
}

} // namespace ptxsim
