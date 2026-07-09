// src/ptxsim/memory/tmem_allocator.cpp
// Phase 1 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q1-A).
//
// Per ptx-lessons-learned §2 (recursive locking), each public method
// holds `mu_` independently and calls only `_*_locked_` private
// helpers while holding the lock. `validate_slot_id_` is a pure
// input check and is safe to call from any context (no lock).

#include "ptxsim/memory/tmem_allocator.h"

#include <stdexcept>
#include <string>

namespace {
inline void throw_error(const std::string& msg) {
    throw std::runtime_error(msg);
}
}  // namespace

TmemAllocator::TmemAllocator() = default;

TmemAllocator::~TmemAllocator() = default;

// ---------------------------------------------------------------------------
// Pure input check — no state access, no lock needed.
// ---------------------------------------------------------------------------
bool TmemAllocator::validate_slot_id_(size_t slot_id) {
    return slot_id < kSlotCount;
}

// ---------------------------------------------------------------------------
// First-fit scan — caller MUST hold `mu_` (reads `allocation_map_`).
// ---------------------------------------------------------------------------
bool TmemAllocator::find_free_range_locked_(size_t num_cols,
                                            size_t& out_start) const {
    if (num_cols == 0 || num_cols > kSlotCount) {
        return false;
    }
    // Find the first run of `num_cols` consecutive 0-bits.
    for (size_t i = 0; i + num_cols <= kSlotCount; ++i) {
        bool run_free = true;
        for (size_t j = 0; j < num_cols; ++j) {
            if (allocation_map_.test(i + j)) {
                run_free = false;
                break;
            }
        }
        if (run_free) {
            out_start = i;
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Mark bits as allocated — caller MUST hold `mu_`.
// ---------------------------------------------------------------------------
void TmemAllocator::mark_allocated_locked_(size_t start, size_t num_cols) {
    for (size_t i = 0; i < num_cols; ++i) {
        allocation_map_.set(start + i);
    }
    allocations_.emplace(start, num_cols);
}

// ---------------------------------------------------------------------------
// Mark bits as free — caller MUST hold `mu_`. Caller is expected to
// have already erased the matching entry from `allocations_` (or to
// call this AFTER the erase — see `deallocate`).
// ---------------------------------------------------------------------------
void TmemAllocator::mark_free_locked_(size_t start, size_t num_cols) {
    for (size_t i = 0; i < num_cols; ++i) {
        allocation_map_.reset(start + i);
    }
}

// ---------------------------------------------------------------------------
// Public mutating API.
// ---------------------------------------------------------------------------

size_t TmemAllocator::allocate(size_t num_cols) {
    if (num_cols == 0) {
        throw_error("TmemAllocator::allocate: num_cols must be > 0");
    }
    if (num_cols > kSlotCount) {
        throw_error("TmemAllocator::allocate: num_cols " +
                    std::to_string(num_cols) + " exceeds kSlotCount " +
                    std::to_string(kSlotCount));
    }

    std::lock_guard<std::mutex> lock(mu_);

    size_t start = kInvalidSlotId;
    if (!find_free_range_locked_(num_cols, start)) {
        return kInvalidSlotId;  // OOM — caller decides fallback
    }
    mark_allocated_locked_(start, num_cols);
    return start;
}

void TmemAllocator::deallocate(size_t slot_id) {
    if (!validate_slot_id_(slot_id)) {
        throw_error("TmemAllocator::deallocate: slot_id " +
                    std::to_string(slot_id) + " out of range [0, " +
                    std::to_string(kSlotCount) + ")");
    }

    std::lock_guard<std::mutex> lock(mu_);

    auto it = allocations_.find(slot_id);
    if (it == allocations_.end()) {
        throw_error("TmemAllocator::deallocate: slot_id " +
                    std::to_string(slot_id) +
                    " is not the start of an active allocation");
    }
    size_t num_cols = it->second;
    allocations_.erase(it);
    mark_free_locked_(slot_id, num_cols);
}

// ---------------------------------------------------------------------------
// Public read-only API — does NOT hold `mu_`. Safe to call from any
// context. Returns a snapshot; concurrent mutations may race but the
// result is well-defined for each individual bit/key.
// ---------------------------------------------------------------------------

bool TmemAllocator::is_allocated_start(size_t slot_id) const {
    if (!validate_slot_id_(slot_id)) {
        return false;
    }
    // std::map::find is safe under concurrent insertion/erasure of
    // *other* keys (node-based container). Reading `count(key)` for
    // a specific key while a different key is being inserted/erased
    // is well-defined per the C++17 standard.
    return allocations_.find(slot_id) != allocations_.end();
}

bool TmemAllocator::is_allocated(size_t slot_id) const {
    if (!validate_slot_id_(slot_id)) {
        return false;
    }
    return allocation_map_.test(slot_id);
}

size_t TmemAllocator::active_allocation_count() const {
    return allocations_.size();
}

size_t TmemAllocator::total_allocated_slots() const {
    size_t total = 0;
    for (const auto& [start, num_cols] : allocations_) {
        total += num_cols;
    }
    return total;
}
