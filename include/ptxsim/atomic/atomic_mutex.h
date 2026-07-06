// atomic_mutex.h
#pragma once

#include <mutex>

namespace ptxsim {

// Single global mutex serializing atomic memory operations across all
// warps and CTAs.
//
// Lock-order proof vs CTABarrier::mutex_ (and BarrierModule::mu_):
//   - processAtomicOperation is the only call site of this mutex.
//   - It never calls into CTABarrier::arrive/reset/is_complete under the
//     lock, and CTABarrier never re-enters atomic handlers while holding
//     its own mutex.
//   - Therefore the two mutexes are NEVER held simultaneously, so the
//     deadlock-by-acquisition-order concern does not apply.
//
// Acquire via AtomicLockGuard (RAII). Released on scope exit.
class AtomicMutex {
public:
    void lock();
    void unlock();
private:
    std::mutex mu_;
};

// Meyers-singleton accessor. Constructed on first use; safe under static
// initialization order fiasco.
AtomicMutex& global_atomic_mutex();

class AtomicLockGuard {
public:
    explicit AtomicLockGuard(AtomicMutex& m);
    ~AtomicLockGuard();

    AtomicLockGuard(const AtomicLockGuard&) = delete;
    AtomicLockGuard& operator=(const AtomicLockGuard&) = delete;
private:
    AtomicMutex& mu_;
};

} // namespace ptxsim
