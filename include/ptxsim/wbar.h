#ifndef WBAR_H
#define WBAR_H

#include <cstdint>
#include <vector>
#include <map>

namespace ptxsim {

// ============================================================================
// DEPRECATED: legacy Wbar struct. Production handlers (BarHandler,
// BarWarpSyncHandler) now use BarrierModule / WarpBarrier from
// include/ptxsim/barrier/. This header is kept temporarily for tests that
// still reference the old API. New code MUST use BarrierModule APIs.
//
// Migration map:
//   Wbar wbar;                              → WarpBarrier* wbar = bm.get_warp_barrier(0);
//   wbar.init(mask, reconv_pc)              → bm.init_warp_barrier(0, mask, reconv_pc, barrier_pc);
//   wbar.arrive(lane_id)                    → bm.arrive_at_warp_barrier(0, lane_id);
//   wbar.is_complete()                      → wbar->is_complete()
//   wbar.count_participants()               → wbar->get_expected_count()
//   wbar.count_arrived()                    → wbar->get_arrived_count()
//   wbar.arrived_mask / participation_mask → wbar->get_arrived_mask() / get_participation_mask()
//   wbar.reset()                            → wbar->reset()
//   warp_state.wbars[0] / current_wbar_id   → bm.get_warp_barrier(0) / wbar->is_initialized()
//   wbar.reconvergence_pc                   → wbar->get_reconvergence_pc()
//
// Removal scheduled after all tests migrate to BarrierModule API.
// ============================================================================
struct [[deprecated("Use ptxsim::WarpBarrier from ptxsim/barrier/barrier_module.h")]] Wbar {
    uint32_t participation_mask = 0;
    uint32_t arrived_mask = 0;
    int reconvergence_pc = -1;
    uint32_t barrier_pc = 0;  // PC of the barrier instruction itself
    bool is_initialized = false;
    int expected_count = 0;
    
    bool memory_fence_verification_enabled = false;
    std::map<int, std::vector<uint64_t>> pre_barrier_stores;
    
    void reset() {
        participation_mask = 0;
        arrived_mask = 0;
        reconvergence_pc = -1;
        barrier_pc = 0;
        is_initialized = false;
        expected_count = 0;
        pre_barrier_stores.clear();
    }
    
    bool is_complete() const {
        if (!is_initialized || participation_mask == 0) {
            return false;
        }
        return (arrived_mask & participation_mask) == participation_mask;
    }
    
    int count_participants() const {
        return __builtin_popcount(participation_mask);
    }
    
    int count_arrived() const {
        return __builtin_popcount(arrived_mask);
    }
    
    void arrive(int lane_id) {
        if (lane_id >= 0 && lane_id < 32) {
            arrived_mask |= (1u << lane_id);
        }
    }
    
    void set_participants(uint32_t mask) {
        participation_mask = mask;
        expected_count = __builtin_popcount(mask);
    }
    
    void set_reconvergence_pc(int pc) {
        reconvergence_pc = pc;
    }
    
    void init(uint32_t participants, int reconvergence_pc) {
        reset();
        participation_mask = participants;
        expected_count = __builtin_popcount(participants);
        this->reconvergence_pc = reconvergence_pc;
        is_initialized = true;
    }
    
#ifdef PTX_DEBUG
    void enable_memory_fence_verification(bool enable) {
        memory_fence_verification_enabled = enable;
    }
    
    bool is_memory_fence_verification_enabled() const {
        return memory_fence_verification_enabled;
    }
    
    void record_pre_barrier_store(int lane_id, uint64_t addr) {
        pre_barrier_stores[lane_id].push_back(addr);
    }
    
    int get_pre_barrier_store_count() const {
        int count = 0;
        for (const auto& kv : pre_barrier_stores) {
            count += kv.second.size();
        }
        return count;
    }
    
    bool verify_all_stores_visible() const {
        if (!memory_fence_verification_enabled) {
            return true;
        }
        return (arrived_mask & participation_mask) == participation_mask;
    }
#endif
};

}

#endif
