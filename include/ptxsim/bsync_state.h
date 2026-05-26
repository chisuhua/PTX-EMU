#ifndef BSYNC_STATE_H
#define BSYNC_STATE_H

#include <cstdint>
#include <unordered_map>

namespace ptxsim {

struct BsyncState {
    uint32_t barrier_id = 0;
    uint32_t waiting_threads_mask = 0;
    uint32_t total_threads = 0;
    uint32_t suspended_pc = 0;
    bool is_released = false;
    
    bool all_arrived() const {
        if (total_threads == 0) return false;
        uint32_t arrived_count = 0;
        uint32_t mask = waiting_threads_mask;
        while (mask) {
            arrived_count += mask & 1;
            mask >>= 1;
        }
        return arrived_count >= total_threads;
    }
    
    void reset() {
        barrier_id = 0;
        waiting_threads_mask = 0;
        total_threads = 0;
        suspended_pc = 0;
        is_released = false;
    }
};

class BsyncManager {
public:
    BsyncManager() = default;
    
    void bssy(uint32_t barrier_id, uint32_t thread_mask);
    bool bsync(uint32_t barrier_id, uint32_t lane_id, uint32_t current_pc);
    bool check_release(uint32_t barrier_id);
    void release(uint32_t barrier_id);
    
    BsyncState* get_state(uint32_t barrier_id);
    const BsyncState* get_state(uint32_t barrier_id) const;
    
    bool is_waiting(uint32_t barrier_id, uint32_t lane_id) const;
    uint32_t get_waiting_mask(uint32_t barrier_id) const;
    
    void cleanup();
    void reset();
    size_t size() const { return barriers_.size(); }

private:
    std::unordered_map<uint32_t, BsyncState> barriers_;
};

enum class DivergenceExecutionMode {
    Sequential,
    Interleaved,
    ShortestFirst
};

inline const char* divergence_execution_mode_to_string(DivergenceExecutionMode mode) {
    switch (mode) {
        case DivergenceExecutionMode::Sequential:    return "sequential";
        case DivergenceExecutionMode::Interleaved:   return "interleaved";
        case DivergenceExecutionMode::ShortestFirst: return "shortest_first";
        default:                                      return "unknown";
    }
}

} // namespace ptxsim

#endif // BSYNC_STATE_H
