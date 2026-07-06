// atomic_mutex.cpp
#include "ptxsim/atomic/atomic_mutex.h"

namespace ptxsim {

AtomicMutex& global_atomic_mutex() {
    static AtomicMutex instance;
    return instance;
}

void AtomicMutex::lock() {
    mu_.lock();
}

void AtomicMutex::unlock() {
    mu_.unlock();
}

AtomicLockGuard::AtomicLockGuard(AtomicMutex& m) : mu_(m) {
    mu_.lock();
}

AtomicLockGuard::~AtomicLockGuard() {
    mu_.unlock();
}

} // namespace ptxsim
