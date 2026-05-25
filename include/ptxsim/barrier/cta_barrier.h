// cta_barrier.h
#ifndef CTA_BARRIER_H
#define CTA_BARRIER_H

#include "barrier_types.h"
#include <cstdint>
#include <set>
#include <mutex>
#include <memory>

namespace ptxsim {

class ThreadContext;

class CTABarrier {
public:
    CTABarrier();
    explicit CTABarrier(int barrier_id);

    void init(int barrier_id, int total_threads, int warp_count);

    bool arrive(ThreadContext* thread);

    bool is_complete() const;

    int get_barrier_id() const { return barrier_id_; }
    int get_expected_threads() const { return expected_threads_; }
    int get_arrived_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(arrived_threads_.size());
    }
    const std::set<ThreadContext*>& get_waiting_threads() const { return arrived_threads_; }
    int get_warp_count() const { return warp_count_; }

    void reset();

#ifdef PTX_DEBUG
    std::string dump() const;
#endif

private:
    int barrier_id_;
    int expected_threads_;
    int warp_count_;
    std::set<ThreadContext*> arrived_threads_;
    mutable std::mutex mutex_;
    bool is_initialized_;
};

} // namespace ptxsim

#endif // CTA_BARRIER_H