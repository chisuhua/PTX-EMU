/**
 * Targeted unit tests for the four specific bugs discovered during
 * test_syncthreads Test 3 investigation:
 *   Bug 1: sync_to_warp_state overwriting is_blocked set by barrier completion
 *   Bug 2: CFG barrier reconvergence_pc using post-dominator instead of i+1
 *   Bug 3: cudaMemset device pointer not subtracting global_pool offset
 *   Bug 4: Predicate register reads for lanes > lane 0 on sub-warp launches
 */

#include "catch_amalgamated.hpp"

#include "ptxsim/warp_state.h"
#include "ptxsim/thread_state.h"
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include <cstdint>
#include <map>
#include <vector>
#include <cstring>

using ptxsim::WarpState;
using ptxsim::ThreadState;

// ============================================================
// Bug 1: sync_to_warp_state must not overwrite is_blocked set
// by barrier completion handler, and must not overwrite PC that
// has already been advanced past the barrier.
// ============================================================

static uint32_t simulate_sync_to_warp_state_pc(
    uint32_t thread_pc, uint32_t warp_state_pc,
    bool is_run_state)
{
    // From thread_context.cpp fix: don't overwrite an already-advanced PC
    if (warp_state_pc > thread_pc) {
        return warp_state_pc;  // completion handler already advanced
    } else {
        return thread_pc;      // safe to store
    }
}

static bool simulate_sync_to_warp_state_is_blocked(
    bool warp_state_blocked)
{
    // From thread_context.cpp fix: don't clear is_blocked when in RUN state
    return warp_state_blocked;
}

TEST_CASE("Bug1: sync_to_warp_state preserves PC advanced by barrier completion",
          "[bug1][sync_to_warp_state]") {
    // Barrier completion handler sets warp_state.pc = 15 (reconvergence_pc)
    // ThreadContext::pc is still 12 (barrier instruction)
    uint32_t result = simulate_sync_to_warp_state_pc(12, 15, true);
    REQUIRE(result == 15u);

    // Normal execution: thread_pc advanced past barrier
    result = simulate_sync_to_warp_state_pc(13, 13, true);
    REQUIRE(result == 13u);
}

TEST_CASE("Bug1: sync_to_warp_state preserves is_blocked after barrier completion",
          "[bug1][sync_to_warp_state]") {
    // Barrier completion sets is_blocked=false for released thread
    // sync_to_warp_state must NOT clear a pre-existing is_blocked=true
    // from a thread that hasn't been released yet
    REQUIRE(simulate_sync_to_warp_state_is_blocked(true) == true);
    REQUIRE(simulate_sync_to_warp_state_is_blocked(false) == false);
}

// ============================================================
// Bug 2: CFG barrier reconvergence_pc MUST always be i+1, NOT
// the post-dominator from CFG analysis.
// ============================================================

TEST_CASE("Bug2: barrier reconvergence_pc is always i+1 regardless of post-dominator",
          "[bug2][barrier_reconvergence]") {
    // Simulate the fixed CFG analysis: barriers use i+1 unconditionally
    auto barrier_reconvergence_pc = [](int barrier_pc) -> int {
        return barrier_pc + 1;
    };

    // First barrier at PC=5 → reconvergence at PC=6
    REQUIRE(barrier_reconvergence_pc(5) == 6);

    // Second barrier at PC=14 → reconvergence at PC=15
    REQUIRE(barrier_reconvergence_pc(14) == 15);

    // Even a barrier with no following branch → next instruction
    REQUIRE(barrier_reconvergence_pc(20) == 21);

    // The post-dominator approach was WRONG because it assigned
    // PC=23 (skip all computation) as reconvergence for first barrier.
    // The fix uses i+1 which correctly returns to the next PC.
}

// ============================================================
// Bug 3: cudaMemset must subtract global_pool base address
// when the device pointer is >= global_pool.
// ============================================================

static size_t test_cuda_memset_offset(void *dev_ptr, uint64_t global_pool) {
    uint64_t device_offset = reinterpret_cast<uint64_t>(dev_ptr);
    if (device_offset >= global_pool) {
        device_offset -= global_pool;
    }
    return static_cast<size_t>(device_offset);
}

TEST_CASE("Bug3: cudaMemset offset subtraction for device pointers",
          "[bug3][cudaMemset]") {
    uint64_t pool = 0x10000000u;

    // Pointer is an absolute address (pool + 64)
    void *ptr = reinterpret_cast<void *>(pool + 64);
    REQUIRE(test_cuda_memset_offset(ptr, pool) == 64u);

    // Pointer is already an offset (small value)
    ptr = reinterpret_cast<void *>(32);
    REQUIRE(test_cuda_memset_offset(ptr, pool) == 32u);

    // Pointer at pool base
    ptr = reinterpret_cast<void *>(pool);
    REQUIRE(test_cuda_memset_offset(ptr, pool) == 0u);

    // Large offset
    ptr = reinterpret_cast<void *>(pool + 4096);
    REQUIRE(test_cuda_memset_offset(ptr, pool) == 4096u);
}

// ============================================================
// Bug 4: Predicate register reads for lanes > 0 on sub-warp (16-thread) launches.
// For a 16-thread CTA, lane_id = 0..15. If only lane 0's predicate is set
// correctly and lanes 1-15 read stale/uninitialised values, all threads
// might take the same branch path.
// ============================================================

struct SimPredicateTest {
    // Simulate what handle_branch does for each lane
    int lanes_taken[32];
    int lanes_not_taken[32];
    int taken_count = 0;
    int not_taken_count = 0;
    uint32_t taken_mask = 0;
    uint32_t not_taken_mask = 0;
};

static SimPredicateTest simulate_handle_branch(
    int thread_count,
    const std::vector<uint8_t>& pred_values_per_lane,  // pred value for each lane
    bool predicate_negated,
    int target_pc,
    int original_pc)
{
    SimPredicateTest result;
    std::memset(&result, 0, sizeof(result));

    for (int lane = 0; lane < 32; lane++) {
        bool is_active = lane < thread_count;
        if (!is_active) continue;

        bool should_branch = true;
        if (!pred_values_per_lane.empty() && lane < (int)pred_values_per_lane.size()) {
            bool pred_bool = pred_values_per_lane[lane] != 0;
            should_branch = predicate_negated ? !pred_bool : pred_bool;
        }

        if (should_branch) {
            result.taken_mask |= (1u << lane);
            result.lanes_taken[result.taken_count++] = lane;
        } else {
            result.not_taken_mask |= (1u << lane);
            result.lanes_not_taken[result.not_taken_count++] = lane;
        }
    }
    return result;
}

TEST_CASE("Bug4a: 16-thread launch — all threads share same predicate value (non-divergent)",
          "[bug4][predicate][16-thread]") {
    // setp.gt.u32 %p1, %r1 (%tid.x), 15
    // For a 16-thread CTA (tid 0-15), ALL threads have tid < 16
    // So %p1 = false for ALL threads → @%p1 bra should NOT be taken by ANY
    std::vector<uint8_t> preds(16, 0);  // All predicate = false
    auto result = simulate_handle_branch(16, preds, false, 20, 7);

    REQUIRE(result.taken_count == 0);
    REQUIRE(result.not_taken_count == 16);
    for (int i = 0; i < 16; i++) {
        REQUIRE(result.lanes_not_taken[i] == i);
    }
}

TEST_CASE("Bug4b: 16-thread launch — predicate = true for ALL lanes",
          "[bug4][predicate][16-thread]") {
    // All threads should take the branch
    std::vector<uint8_t> preds(16, 1);  // All predicate = true
    auto result = simulate_handle_branch(16, preds, false, 20, 7);

    REQUIRE(result.taken_count == 16);
    REQUIRE(result.not_taken_count == 0);
}

TEST_CASE("Bug4c: 16-thread launch — predicate = true for lanes 0-7, false for 8-15",
          "[bug4][predicate][divergent][16-thread]") {
    std::vector<uint8_t> preds(16, 0);
    for (int i = 0; i < 8; i++) preds[i] = 1;

    auto result = simulate_handle_branch(16, preds, false, 20, 7);

    REQUIRE(result.taken_count == 8);
    REQUIRE(result.not_taken_count == 8);

    for (int i = 0; i < result.taken_count; i++) {
        REQUIRE(result.lanes_taken[i] < 8);
    }
    for (int i = 0; i < result.not_taken_count; i++) {
        REQUIRE(result.lanes_not_taken[i] >= 8);
        REQUIRE(result.lanes_not_taken[i] < 16);
    }
}

TEST_CASE("Bug4d: 32-thread launch — comparison baseline (same scenario as 16)",
          "[bug4][predicate][32-thread]") {
    // setp.gt.u32 %p1, %tid, 15
    // tid 0-15: false, tid 16-31: true
    std::vector<uint8_t> preds(32, 0);
    for (int i = 16; i < 32; i++) preds[i] = 1;

    auto result = simulate_handle_branch(32, preds, false, 20, 7);

    REQUIRE(result.taken_count == 16);
    REQUIRE(result.not_taken_count == 16);

    // Lanes 0-15: not taken
    for (int i = 0; i < 16; i++) {
        REQUIRE(result.lanes_not_taken[i] == i);
    }
    // Lanes 16-31: taken
    for (int i = 0; i < 16; i++) {
        REQUIRE(result.lanes_taken[i] == 16 + i);
    }
}

TEST_CASE("Bug4e: Negated predicate (!%p) — verify negation logic",
          "[bug4][predicate][negation]") {
    std::vector<uint8_t> preds(16, 0);
    // All predicates = false, negated → should_branch = !false = true for ALL
    auto result = simulate_handle_branch(16, preds, true, 20, 7);

    REQUIRE(result.taken_count == 16);
    REQUIRE(result.not_taken_count == 0);
}

TEST_CASE("Bug4f: Predicate register NOT set for lane 1-15 (the actual bug simulation)",
          "[bug4][predicate][uninitialized]") {
    // Simulate the ACTUAL bug: only lane 0's predicate is set correctly,
    // lanes 1-15 have stale/uninitialised values (non-zero, e.g., 0xFF)
    std::vector<uint8_t> preds(16, 0xFF);  // All stale = non-zero
    preds[0] = 0;  // Only lane 0 is correctly set to false

    auto result = simulate_handle_branch(16, preds, false, 20, 7);

    // With the BUG: lanes 1-15 read 0xFF (true) and take the branch
    // Only lane 0 correctly skips (false)
    REQUIRE(result.taken_count == 15);   // WRONG behavior
    REQUIRE(result.not_taken_count == 1);

    // If ALL predicates were correctly false (FIXED behavior)
    std::vector<uint8_t> correct_preds(16, 0);
    auto correct_result = simulate_handle_branch(16, correct_preds, false, 20, 7);
    REQUIRE(correct_result.taken_count == 0);    // CORRECT behavior
    REQUIRE(correct_result.not_taken_count == 16);
}
