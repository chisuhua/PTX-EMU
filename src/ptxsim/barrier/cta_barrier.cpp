// cta_barrier.cpp
#include "ptxsim/barrier/cta_barrier.h"
#include "ptxsim/thread_context.h"
#include "utils/logger.h"

namespace ptxsim {

CTABarrier::CTABarrier()
    : barrier_id_(0), expected_threads_(0), warp_count_(0), is_initialized_(false) {
}

CTABarrier::CTABarrier(int barrier_id)
    : barrier_id_(barrier_id), expected_threads_(0), warp_count_(0), is_initialized_(false) {
}

void CTABarrier::init(int barrier_id, int total_threads, int warp_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    barrier_id_ = barrier_id;
    expected_threads_ = total_threads;
    warp_count_ = warp_count;
    arrived_threads_.clear();
    is_initialized_ = true;

    PTX_DEBUG_EMU("CTABarrier::init id=%d threads=%d warps=%d",
                  barrier_id, total_threads, warp_count);
}

bool CTABarrier::arrive(ThreadContext* thread) {
    if (!is_initialized_) {
        PTX_ERROR_EMU("CTABarrier::arrive called on uninitialized barrier");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (arrived_threads_.find(thread) != arrived_threads_.end()) {
        PTX_DEBUG_EMU("CTABarrier::arrive thread already waiting, skipping");
        return false;
    }

    arrived_threads_.insert(thread);

    PTX_DEBUG_EMU("CTABarrier::arrive id=%d arrived=%d/%d",
                  barrier_id_, arrived_threads_.size(), expected_threads_);

    if (is_complete()) {
        PTX_INFO_EMU("CTABarrier::complete id=%d threads=%d",
                     barrier_id_, arrived_threads_.size());
        return true;
    }

    return false;
}

bool CTABarrier::is_complete() const {
    if (!is_initialized_) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    return arrived_threads_.size() >= static_cast<size_t>(expected_threads_);
}

void CTABarrier::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    barrier_id_ = 0;
    expected_threads_ = 0;
    warp_count_ = 0;
    arrived_threads_.clear();
    is_initialized_ = false;
}

#ifdef PTX_DEBUG
std::string CTABarrier::dump() const {
    std::lock_guard<std::mutex> lock(mutex_);
    char buf[256];
    snprintf(buf, sizeof(buf),
             "CTABarrier id=%d expected=%d arrived=%zu warps=%d",
             barrier_id_, expected_threads_, arrived_threads_.size(), warp_count_);
    return std::string(buf);
}
#endif

} // namespace ptxsim