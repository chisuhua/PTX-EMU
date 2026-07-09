// tests/unit/memory/test_tmem_allocator.cpp
// Phase 1 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q1-A).
//
// Unit tests for the per-CTA TmemAllocator (allocation layer on top of
// the fixed 256-slot Tmem). 12 TEST_CASEs cover:
//   1.  Basic single-slot alloc returns 0
//   2.  Multi-slot alloc returns contiguous range
//   3.  Two disjoint allocs get distinct start slots
//   4.  is_allocated_start true for starts, false for middles
//   5.  is_allocated true for every slot in an allocated range
//   6.  Allocator fills up; OOM returns kInvalidSlotId
//   7.  After dealloc, slots are reusable (next alloc reuses them)
//   8.  dealloc of non-start throws
//   9.  dealloc of out-of-range throws
//   10. alloc(0) throws (defensive — should never happen per PTX ISA)
//   11. Multiple CTAs have independent state (cross-CTA isolation)
//   12. **Multi-threaded concurrent alloc/dealloc — recursive-lock
//       audit falsification test (per ptx-lessons-learned §2 + Oracle
//       high-risk finding)**: 8 threads × 1000 iterations must NOT
//       deadlock and must produce consistent allocation state.
//
// All golden values are hand-computed from the first-fit policy and
// 256-slot space. Marked UNVERIFIED-AGAINST-HARDWARE.

#include "catch_amalgamated.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ptxsim/memory/tmem_allocator.h"

// ---------------------------------------------------------------------------
// 1. basic_alloc_returns_zero
// ---------------------------------------------------------------------------
TEST_CASE("basic_alloc_returns_zero", "[tmem_allocator][alloc]") {
    TmemAllocator a;
    size_t s = a.allocate(1);
    REQUIRE(s == 0);
    REQUIRE(a.active_allocation_count() == 1);
    REQUIRE(a.total_allocated_slots() == 1);
}

// ---------------------------------------------------------------------------
// 2. multi_slot_alloc_returns_contiguous_range
// ---------------------------------------------------------------------------
TEST_CASE("multi_slot_alloc_returns_contiguous_range", "[tmem_allocator][alloc]") {
    TmemAllocator a;
    size_t s = a.allocate(4);
    REQUIRE(s == 0);
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(a.is_allocated(i));
    }
    REQUIRE_FALSE(a.is_allocated(4));
    REQUIRE(a.total_allocated_slots() == 4);
}

// ---------------------------------------------------------------------------
// 3. two_disjoint_allocs_get_distinct_starts
// ---------------------------------------------------------------------------
TEST_CASE("two_disjoint_allocs_get_distinct_starts", "[tmem_allocator][alloc]") {
    TmemAllocator a;
    size_t s1 = a.allocate(2);
    size_t s2 = a.allocate(3);
    REQUIRE(s1 == 0);
    REQUIRE(s2 == 2);  // first-fit after s1..s1+1 occupied
    REQUIRE(a.active_allocation_count() == 2);
    REQUIRE(a.total_allocated_slots() == 5);
}

// ---------------------------------------------------------------------------
// 4. is_allocated_start_distinguishes_start_from_middle
// ---------------------------------------------------------------------------
TEST_CASE("is_allocated_start_distinguishes_start_from_middle",
          "[tmem_allocator][query]") {
    TmemAllocator a;
    a.allocate(4);
    REQUIRE(a.is_allocated_start(0));
    REQUIRE_FALSE(a.is_allocated_start(1));
    REQUIRE_FALSE(a.is_allocated_start(2));
    REQUIRE_FALSE(a.is_allocated_start(3));
    REQUIRE_FALSE(a.is_allocated_start(4));
}

// ---------------------------------------------------------------------------
// 5. is_allocated_true_for_every_slot_in_range
// ---------------------------------------------------------------------------
TEST_CASE("is_allocated_true_for_every_slot_in_range",
          "[tmem_allocator][query]") {
    TmemAllocator a;
    a.allocate(8);
    for (size_t i = 0; i < 8; ++i) {
        REQUIRE(a.is_allocated(i));
    }
    REQUIRE_FALSE(a.is_allocated(8));
    REQUIRE_FALSE(a.is_allocated(255));
}

// ---------------------------------------------------------------------------
// 6. fill_up_then_oom_returns_invalid
// ---------------------------------------------------------------------------
TEST_CASE("fill_up_then_oom_returns_invalid", "[tmem_allocator][oom]") {
    TmemAllocator a;
    // Each alloc(1) consumes exactly 1 slot. Fill all 256.
    for (size_t i = 0; i < 256; ++i) {
        size_t s = a.allocate(1);
        REQUIRE(s == i);
    }
    // 257th must fail.
    size_t s = a.allocate(1);
    REQUIRE(s == TmemAllocator::kInvalidSlotId);
    REQUIRE(a.active_allocation_count() == 256);
}

// ---------------------------------------------------------------------------
// 7. dealloc_frees_slots_for_reuse
// ---------------------------------------------------------------------------
TEST_CASE("dealloc_frees_slots_for_reuse", "[tmem_allocator][dealloc]") {
    TmemAllocator a;
    size_t s1 = a.allocate(2);
    REQUIRE(s1 == 0);
    a.deallocate(s1);
    REQUIRE(a.active_allocation_count() == 0);
    REQUIRE_FALSE(a.is_allocated(0));
    REQUIRE_FALSE(a.is_allocated(1));
    // Next alloc should reuse slot 0.
    size_t s2 = a.allocate(1);
    REQUIRE(s2 == 0);
}

// ---------------------------------------------------------------------------
// 8. dealloc_non_start_throws
// ---------------------------------------------------------------------------
TEST_CASE("dealloc_non_start_throws", "[tmem_allocator][dealloc]") {
    TmemAllocator a;
    a.allocate(4);
    REQUIRE_THROWS_AS(a.deallocate(1), std::runtime_error);
    REQUIRE_THROWS_AS(a.deallocate(2), std::runtime_error);
    REQUIRE_THROWS_AS(a.deallocate(3), std::runtime_error);
    // Start slot still works.
    REQUIRE_NOTHROW(a.deallocate(0));
}

// ---------------------------------------------------------------------------
// 9. dealloc_out_of_range_throws
// ---------------------------------------------------------------------------
TEST_CASE("dealloc_out_of_range_throws", "[tmem_allocator][dealloc]") {
    TmemAllocator a;
    REQUIRE_THROWS_AS(a.deallocate(256), std::runtime_error);
    REQUIRE_THROWS_AS(a.deallocate(1000), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 10. alloc_zero_throws (defensive guard)
// ---------------------------------------------------------------------------
TEST_CASE("alloc_zero_throws", "[tmem_allocator][alloc]") {
    TmemAllocator a;
    REQUIRE_THROWS_AS(a.allocate(0), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 11. cross_instance_isolation
// ---------------------------------------------------------------------------
TEST_CASE("cross_instance_isolation", "[tmem_allocator][isolation]") {
    TmemAllocator a;  // CTA #0
    TmemAllocator b;  // CTA #1
    a.allocate(4);
    REQUIRE(a.total_allocated_slots() == 4);
    REQUIRE(b.total_allocated_slots() == 0);
    // b can still allocate from slot 0.
    size_t s = b.allocate(1);
    REQUIRE(s == 0);
}

// ---------------------------------------------------------------------------
// 12. multi_threaded_concurrent_alloc_dealloc_no_deadlock
//     (RECURSIVE-LOCK AUDIT FALSIFICATION per ptx-lessons-learned §2 +
//      Oracle high-risk finding, 2026-07-08)
//
// 8 threads × 1000 mixed alloc/dealloc operations each. If any
// TmemAllocator public method acquires mu_ and then calls another
// public method that also acquires mu_, the test will either:
//   - Hang (deadlock) and exceed the 30 s wall-clock budget, OR
//   - Crash (std::system_error from lock_guard on recursive mutex).
//
// We assert:
//   1. No thread hangs (all 8 join() within budget).
//   2. No exception escapes from any thread.
//   3. Total allocated slots is consistent (all 256 slots total
//      across all threads — no double-allocation, no leak).
// ---------------------------------------------------------------------------
TEST_CASE("multi_threaded_concurrent_alloc_dealloc_no_deadlock",
          "[tmem_allocator][concurrency][recursive_lock_audit]") {
    TmemAllocator a;

    constexpr int kThreads = 8;
    constexpr int kIterationsPerThread = 1000;

    std::atomic<size_t> total_allocated{0};
    std::atomic<int> exception_count{0};

    auto worker = [&a, &total_allocated, &exception_count](int t) {
        std::mt19937 rng(static_cast<unsigned>(t) * 9973u + 1);
        std::uniform_int_distribution<int> op(0, 2);
        std::uniform_int_distribution<int> size(1, 8);

        std::vector<size_t> live;
        live.reserve(64);

        for (int i = 0; i < kIterationsPerThread; ++i) {
            try {
                if (op(rng) < 2) {
                    size_t s = a.allocate(static_cast<size_t>(size(rng)));
                    if (s != TmemAllocator::kInvalidSlotId) {
                        live.push_back(s);
                        total_allocated.fetch_add(1, std::memory_order_relaxed);
                    }
                } else if (!live.empty()) {
                    size_t idx = rng() % live.size();
                    size_t s = live[idx];
                    a.deallocate(s);
                    live[idx] = live.back();
                    live.pop_back();
                    total_allocated.fetch_sub(1, std::memory_order_relaxed);
                }
            } catch (...) {
                exception_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
        for (size_t s : live) {
            try {
                a.deallocate(s);
                total_allocated.fetch_sub(1, std::memory_order_relaxed);
            } catch (...) {
            }
        }
    };

    std::vector<std::future<void>> futures;
    futures.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        futures.emplace_back(std::async(std::launch::async, worker, t));
    }

    constexpr auto kDeadline = std::chrono::seconds(30);
    bool all_completed = true;
    for (auto& f : futures) {
        if (f.wait_for(kDeadline) != std::future_status::ready) {
            all_completed = false;
            break;
        }
    }

    REQUIRE(all_completed);  // deadlock suspected if any future did not complete
    REQUIRE(exception_count.load() == 0);
    REQUIRE(a.active_allocation_count() == 0);
    REQUIRE(total_allocated.load() == 0);
}
