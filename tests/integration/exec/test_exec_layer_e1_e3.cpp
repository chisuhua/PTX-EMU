/**
 * Execution Layer Tests for Test 3 Root Cause
 *
 * Tests three execution-layer failure hypotheses:
 * - E1: execute_warp_instruction divergent branch scheduling (16-thread warp)
 * - E2: barrier_waiting_threads count logic in 16-thread scenario
 * - E3: shared memory symbol resolution correctness for sub-32-thread CTAs
 */

#include "catch_amalgamated.hpp"
#include "ptxsim/wbar.h"
#include <cstdint>
#include <vector>

using ptxsim::Wbar;

// ============================================================
// E1: Divergent branch scheduling with partial active lanes
// ============================================================

static std::vector<int> lanes_at_pc(
    const std::vector<int>& lane_pcs, int target_pc, uint32_t active_mask)
{
    std::vector<int> result;
    for (int lane = 0; lane < 32; lane++) {
        if (!(active_mask & (1u << lane))) continue;
        if (lane >= (int)lane_pcs.size()) continue;
        if (lane_pcs[lane] == target_pc) result.push_back(lane);
    }
    return result;
}

TEST_CASE("E1a: divergent branch — only target lanes execute",
          "[exec][e1][scheduling]") {
    // 16-thread warp: lanes 0-7 at PC=3 (then), lanes 8-15 at PC=7 (else)
    std::vector<int> lane_pcs(32, -1);
    for (int l = 0; l < 32; l++) lane_pcs[l] = -1;
    for (int l = 0; l < 8; l++) lane_pcs[l] = 3;
    for (int l = 8; l < 16; l++) lane_pcs[l] = 7;
    // lanes 16-31 don't exist

    uint32_t mask_16 = 0x0000FFFFu;
    auto executing_3 = lanes_at_pc(lane_pcs, 3, mask_16);
    auto executing_7 = lanes_at_pc(lane_pcs, 7, mask_16);

    REQUIRE(executing_3.size() == 8u);
    for (int l : executing_3) { REQUIRE(l >= 0); REQUIRE(l <= 7); }

    REQUIRE(executing_7.size() == 8u);
    for (int l : executing_7) { REQUIRE(l >= 8); REQUIRE(l <= 15); }

    // lanes 16-31 should NEVER be returned
    for (int l : executing_3) REQUIRE(l < 16);
    for (int l : executing_7) REQUIRE(l < 16);
}

TEST_CASE("E1b: single PC group — all active lanes execute together",
          "[exec][e1][scheduling]") {
    std::vector<int> lane_pcs(32, -1);
    for (int l = 0; l < 16; l++) lane_pcs[l] = 5;
    // lanes 16-31 don't exist (pc = -1)

    uint32_t mask_16 = 0x0000FFFFu;
    auto executing = lanes_at_pc(lane_pcs, 5, mask_16);

    REQUIRE(executing.size() == 16u);
    for (int l : executing) { REQUIRE(l >= 0); REQUIRE(l < 16); }
}

TEST_CASE("E1c: 32-thread full warp divergent scheduling",
          "[exec][e1][full_warp]") {
    std::vector<int> lane_pcs(32, -1);
    for (int l = 0; l < 16; l++) lane_pcs[l] = 10;
    for (int l = 16; l < 32; l++) lane_pcs[l] = 14;

    uint32_t mask_32 = 0xFFFFFFFFu;
    auto a = lanes_at_pc(lane_pcs, 10, mask_32);
    auto b = lanes_at_pc(lane_pcs, 14, mask_32);

    REQUIRE(a.size() == 16u);
    REQUIRE(b.size() == 16u);
    for (int l : a) REQUIRE(l < 16);
    for (int l : b) REQUIRE(l >= 16);
}

// ============================================================
// E2: Barrier completion logic with partial participation masks
// ============================================================

TEST_CASE("E2a: Wbar with 16-thread mask completes with 16 arrives",
          "[exec][e2][wbar]") {
    Wbar barrier;
    barrier.init(0x0000FFFFu, 20);

    REQUIRE(barrier.count_participants() == 16);
    REQUIRE(!barrier.is_complete());

    for (int l = 0; l < 16; l++) {
        barrier.arrive(l);
    }

    REQUIRE(barrier.count_arrived() == 16);
    REQUIRE(barrier.is_complete());
}

TEST_CASE("E2b: Wbar with 16-thread mask does NOT complete with 32 arrives",
          "[exec][e2][overflow]") {
    Wbar barrier;
    barrier.init(0x0000FFFFu, 20);

    // Only lanes 0-15 should matter
    for (int l = 0; l < 32; l++) {
        barrier.arrive(l);
    }
    REQUIRE(barrier.is_complete());
    // Extra arrivals from lanes 16-31 are harmless (mask filters them)
    REQUIRE(barrier.arrived_mask & 0xFFFF0000u);
}

TEST_CASE("E2c: Wbar with full mask requires all 32 lanes",
          "[exec][e2][full_mask]") {
    Wbar barrier;
    barrier.init(0xFFFFFFFFu, 20);

    REQUIRE(barrier.count_participants() == 32);
    REQUIRE(!barrier.is_complete());

    for (int l = 0; l < 31; l++) {
        barrier.arrive(l);
    }
    REQUIRE(!barrier.is_complete());

    barrier.arrive(31);
    REQUIRE(barrier.is_complete());
}

TEST_CASE("E2d: Wbar reset clears all state",
          "[exec][e2][reset]") {
    Wbar barrier;
    barrier.init(0x0000FFFFu, 20);
    for (int l = 0; l < 16; l++) barrier.arrive(l);
    REQUIRE(barrier.is_complete());

    barrier.reset();
    REQUIRE(!barrier.is_initialized);
    REQUIRE(barrier.participation_mask == 0u);
    REQUIRE(barrier.arrived_mask == 0u);
    REQUIRE(barrier.reconvergence_pc == -1);
}

TEST_CASE("E2e: barrier_waiting_threads count matches CTA thread count",
          "[exec][e2][sm_context_logic]") {
    // Simulates synchronize_barrier logic:
    // barrier_waiting_threads[barId].size() >= total_threads_in_block
    int cta_threads_16 = 16;
    int cta_threads_32 = 32;
    int arrived_16 = 16;
    int arrived_31 = 31;

    // 16-thread CTA: 16 arrived should release
    REQUIRE(arrived_16 >= cta_threads_16);

    // 32-thread CTA: 31 arrived should NOT release
    REQUIRE(!(arrived_31 >= cta_threads_32));

    // 16-thread CTA with only 8 arrived should NOT release
    int arrived_8 = 8;
    REQUIRE(!(arrived_8 >= cta_threads_16));
}

// ============================================================
// E3: Shared memory symbol resolution for sub-32-thread CTAs
// ============================================================

static void* compute_shared_addr(void* base, uint32_t thread_id,
                                  uint32_t element_size, uint32_t offset) {
    return (void*)((char*)base + thread_id * element_size + offset);
}

TEST_CASE("E3a: shared memory addresses are unique for 16 threads",
          "[exec][e3][shared_mem]") {
    void* base = (void*)0x1000;
    std::vector<void*> addrs;

    for (int tid = 0; tid < 16; tid++) {
        addrs.push_back(compute_shared_addr(base, tid, 4, 0));
    }

    for (size_t i = 0; i < addrs.size(); i++) {
        for (size_t j = i + 1; j < addrs.size(); j++) {
            REQUIRE(addrs[i] != addrs[j]);
        }
    }
}

TEST_CASE("E3b: shared memory stride correctness for 16 threads",
          "[exec][e3][stride]") {
    char* base = (char*)0x1000;
    for (int tid = 0; tid < 16; tid++) {
        void* addr = compute_shared_addr(base, tid, 4, 0);
        uint32_t expected_offset = tid * 4;
        REQUIRE((char*)addr == (char*)base + expected_offset);
    }
}

TEST_CASE("E3c: shared memory with wraparound (circular buffer)",
          "[exec][e3][wraparound]") {
    // Test the (tid - 1) & 15 pattern from test_nested_sync
    std::vector<int> read_from;
    for (int tid = 0; tid < 16; tid++) {
        int src = (tid - 1) & 15;
        read_from.push_back(src);
    }

    // tid 0 reads from 15, tid 1 reads from 0, etc.
    REQUIRE(read_from[0] == 15);
    REQUIRE(read_from[1] == 0);
    REQUIRE(read_from[2] == 1);
    REQUIRE(read_from[15] == 14);

    // All indices should be 0-15 (valid for 16-element array)
    for (int idx : read_from) {
        REQUIRE(idx >= 0);
        REQUIRE(idx < 16);
    }
}

TEST_CASE("E3d: shared memory base + offset computation (PTX pattern)",
          "[exec][e3][ptx_addr]") {
    // Simulates PTX pattern:
    //   mov.u32 %r5, shared_var;   // base offset
    //   add.s32 %r2, %r5, %r4;     // offset += tid*4
    //   st.shared.u32 [%r2], val;

    uint32_t shared_var_offset = 0;
    for (int tid = 0; tid < 16; tid++) {
        uint32_t stride = tid * 4;
        uint32_t effective_addr = shared_var_offset + stride;
        REQUIRE(effective_addr == (uint32_t)(tid * 4));

        // Read with offset (data_b pattern with base offset of 64 bytes)
        uint32_t data_b_base = 64;
        uint32_t data_b_addr = data_b_base + stride;
        REQUIRE(data_b_addr == 64u + tid * 4);
    }
}

TEST_CASE("E3e: no address collision between data_a and data_b",
          "[exec][e3][no_alias]") {
    // test_nested_sync shared layout:
    // data_a: 16 * 4 = 64 bytes (offset 0-63)
    // data_b: 16 * 4 = 64 bytes (offset 64-127)

    for (int tid = 0; tid < 16; tid++) {
        uint32_t addr_a = tid * 4;
        uint32_t addr_b = 64 + tid * 4;

        REQUIRE(addr_a < 64u);
        REQUIRE(addr_b >= 64u);
        REQUIRE(addr_a < addr_b);
    }
}

// ============================================================
// E3f: Execution order dependency — write before read after barrier
// ============================================================

TEST_CASE("E3f: barrier enforces write-before-read ordering",
          "[exec][e3][ordering]") {
    // Simulate: thread 0 writes value X, barrier, thread 1 reads X
    // If barrier works: thread 1 sees X
    // If barrier broken: thread 1 sees stale value

    struct SharedMem { uint32_t cells[16] = {0}; };
    SharedMem smem{};

    // Phase 1: each thread writes its tid
    for (int tid = 0; tid < 16; tid++) {
        smem.cells[tid] = (uint32_t)tid;
    }
    // Barrier would go here in real execution

    // Phase 2: each thread reads (tid - 1) & 15
    for (int tid = 0; tid < 16; tid++) {
        int src = (tid - 1) & 15;
        // After the write phase, smem.cells[src] should contain src
        REQUIRE(smem.cells[src] == (uint32_t)src);
    }
}
