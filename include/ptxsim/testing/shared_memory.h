// shared_memory.h
#ifndef PTXSIM_TESTING_SHARED_MEMORY_H
#define PTXSIM_TESTING_SHARED_MEMORY_H

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ptxsim::testing {

// ============================================================================
// Shared Memory Helpers
// ============================================================================

inline void write_shared(void* base, size_t offset, uint32_t val) {
    static_cast<uint32_t*>(base)[offset] = val;
}

inline uint32_t read_shared(void* base, size_t offset) {
    return static_cast<uint32_t*>(base)[offset];
}

inline void* allocate_shared(size_t elems) {
    void* p = malloc(elems * sizeof(uint32_t));
    memset(p, 0, elems * sizeof(uint32_t));
    return p;
}

} // namespace ptxsim::testing

#endif // PTXSIM_TESTING_SHARED_MEMORY_H