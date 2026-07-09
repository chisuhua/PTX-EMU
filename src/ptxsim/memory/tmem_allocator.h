// src/ptxsim/memory/tmem_allocator.h
// Phase 1 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q1-A).
//
// Per-CTA slot allocator for Blackwell Tensor Memory (TMEM).
// Wraps the existing fixed `Tmem` (256 slot × 128 byte) with explicit
// allocation/deallocation semantics required by `tcgen05.alloc` /
// `tcgen05.dealloc` (PTX ISA §9.7.16).
//
// Why a separate layer (vs. extending `Tmem` directly):
//   - `Tmem` is a passive data store (read/write/clear) used by 5 core
//     handlers (mma/ld/st/commit/wait) which encode slot_ids in fragment
//     layout (per `tcgen05.cpp:332-335`). Replacing it would break
//     those handlers.
//   - The new alloc/dealloc/relinquish handlers need **explicit
//     allocation state** (which slots are owned by which allocation)
//     that does not exist in the raw storage. Adding it as a new
//     abstraction keeps the existing 5 core handlers unchanged.
//
// Recursive-lock safety (per ptx-lessons-learned §2):
//   - `mu_` is a `std::mutex` (NOT recursive).
//   - Mutating public methods (`allocate`, `deallocate`) hold `mu_` via
//     `lock_guard` and only invoke private `_*_locked_` helpers.
//   - `validate_slot_id_` is a pure read-only check; it does NOT hold
//     `mu_` and is safe to call from any context.
//   - Public methods NEVER call other public methods while holding
//     `mu_` (no public→public nested lock acquisition → no deadlock).
//
// `cta_group::2` support (per Oracle Q2-A):
//   - This layer is per-CTA by design (one instance per CTAContext).
//   - `.cta_group::2` cross-CTA allocation tracking is OUT OF SCOPE
//     (deferred to `implement-cta-group-2-dist-smem`, ADR-0018).
//   - The handlers that wrap this class throw `UnsupportedInstructionException`
//     for `.cta_group::2`; this layer itself does not know about cta_group.

#ifndef PTXSIM_MEMORY_TMEM_ALLOCATOR_H
#define PTXSIM_MEMORY_TMEM_ALLOCATOR_H

#include <bitset>
#include <cstddef>
#include <map>
#include <mutex>

class TmemAllocator {
public:
    // Mirrors Tmem::kSlotCount — kept as a separate constant to avoid
    // a hard include dependency on `ptxsim/memory/tmem.h` here.
    static constexpr size_t kSlotCount = 256;

    // Sentinel for allocation failure (no contiguous `num_cols` free range).
    static constexpr size_t kInvalidSlotId = static_cast<size_t>(-1);

    TmemAllocator();
    ~TmemAllocator();

    // Disable copy/move (would corrupt `mu_` and `allocation_map_` invariants).
    TmemAllocator(const TmemAllocator&) = delete;
    TmemAllocator& operator=(const TmemAllocator&) = delete;
    TmemAllocator(TmemAllocator&&) = delete;
    TmemAllocator& operator=(TmemAllocator&&) = delete;

    // -----------------------------------------------------------------------
    // Public mutating API (each holds `mu_`).
    // -----------------------------------------------------------------------

    // Allocates `num_cols` consecutive free slots. Returns the start
    // slot_id, or `kInvalidSlotId` on failure (num_cols == 0, or
    // insufficient contiguous free slots).
    //
    // First-fit allocation policy: scan allocation_map_ from slot 0
    // and return the first slot whose next `num_cols` bits are all 0.
    // Simple, deterministic, and good enough for the 256-slot
    // TMEM space (defragmentation is not required by PTX ISA).
    size_t allocate(size_t num_cols);

    // Releases the allocation that starts at `slot_id`. Throws
    // `std::runtime_error` if `slot_id` is not the start of an
    // active allocation (caller should consult `is_allocated_start`).
    void deallocate(size_t slot_id);

    // -----------------------------------------------------------------------
    // Public read-only API (does NOT hold `mu_`; safe from any context).
    // -----------------------------------------------------------------------

    // True if `slot_id` is the start of an active allocation. Pure
    // read of `allocations_`; does NOT consult `allocation_map_` so
    // it works even for slots that are "in the middle" of an allocation.
    bool is_allocated_start(size_t slot_id) const;

    // True if `slot_id` is allocated (as start OR middle of a range).
    // Pure read of `allocation_map_`.
    bool is_allocated(size_t slot_id) const;

    // Number of currently active allocations (for testing/metrics).
    size_t active_allocation_count() const;

    // Total number of allocated slots (for testing/metrics).
    size_t total_allocated_slots() const;

private:
    // -----------------------------------------------------------------------
    // Private helpers — naming convention: `_locked_` suffix means
    // "MUST be called with `mu_` held"; no suffix means "no lock
    // required" (pure computation on inputs only).
    // -----------------------------------------------------------------------

    // Pure input check, no state access. Safe from any context.
    static bool validate_slot_id_(size_t slot_id);

    // Scan `allocation_map_` for the first run of `num_cols`
    // consecutive 0-bits. Caller MUST hold `mu_`.
    bool find_free_range_locked_(size_t num_cols, size_t& out_start) const;

    // Mark bits [start, start+num_cols) in `allocation_map_` as
    // allocated and record the range in `allocations_`. Caller
    // MUST hold `mu_`.
    void mark_allocated_locked_(size_t start, size_t num_cols);

    // Inverse of `mark_allocated_locked_`. Caller MUST hold `mu_`.
    void mark_free_locked_(size_t start, size_t num_cols);

    // -----------------------------------------------------------------------
    // State.
    // -----------------------------------------------------------------------

    // 1 bit per TMEM slot. 1 = allocated, 0 = free.
    std::bitset<kSlotCount> allocation_map_;

    // Map from start slot_id to the size of that allocation. The
    // presence of a key indicates the start is allocated. Iteration
    // over this map during `deallocate` gives the range to free.
    std::map<size_t, size_t> allocations_;

    // Single mutex protecting both containers. `std::mutex` (not
    // recursive) — recursive acquisition would deadlock per
    // ptx-lessons-learned §2.
    mutable std::mutex mu_;
};

#endif  // PTXSIM_MEMORY_TMEM_ALLOCATOR_H
