// src/ptxsim/memory/tmem.h
// Phase 0.2 (Fix #6): Blackwell per-CTA Tensor Memory (TMEM).
//
// 256 slot × 128 byte = 32 KB per CTA (parallel to shared memory).
// PTX ISA §9.7.13 defines TMEM as a per-CTA scratchpad for tensor
// operations; consumed by tcgen05.* load/store in Phase 1-3.
//
// Design:
//   - std::array<uint8_t, 32*1024> backing store (zeroed at construction)
//   - per-CTA isolation: each CTA gets its own instance; writes do not
//     propagate across instances
//   - std::mutex for thread-safety (Phase 1-3 may invoke from multiple warps).
//     Per ptx-lessons-learned §2 (recursive locking), public methods hold
//     the mutex and no public method calls another public method.
//   - Error rejection: slot_id ≥ 256 or size > 128 throws std::runtime_error

#ifndef PTXSIM_MEMORY_TMEM_H
#define PTXSIM_MEMORY_TMEM_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

class Tmem {
public:
    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    static constexpr size_t kSlotCount = 256;
    static constexpr size_t kSlotSize = 128;
    static constexpr size_t kTotalSize = kSlotCount * kSlotSize;

    Tmem();
    ~Tmem();

    void read(size_t slot_id, void* bytes, size_t size) const;
    void write(size_t slot_id, const void* bytes, size_t size);
    void clear();

    bool validate_slot_id(size_t slot_id) const;

private:
    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    std::array<uint8_t, kTotalSize> storage_;

    // per ptx-lessons-learned §2: no public method calls another public
    // method while holding mu_ (recursive lock → deadlock)
    mutable std::mutex mu_;
};

#endif  // PTXSIM_MEMORY_TMEM_H