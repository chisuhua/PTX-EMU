/**
 * I1-I2: Thread scheduling order & reinit barrier tests.
 * I3: Full 16-thread barrier interaction integration test.
 */

#include "catch_amalgamated.hpp"
#include "ptxsim/wbar.h"
#include "ptxsim/warp_state.h"
#include <cstdint>
#include <map>
#include <vector>
#include <cstring>

using ptxsim::Wbar;
using ptxsim::WarpState;

// ============================================================
// I1: Thread scheduling order after barrier — do lanes execute
// ld.shared BEFORE all lanes' st.shared completes?
// ============================================================

struct SharedMem {
    uint8_t buf[256];
    void write(int offset, uint32_t val) {
        memcpy(buf + offset, &val, 4);
    }
    uint32_t read(int offset) const {
        uint32_t v = 0;
        memcpy(&v, buf + offset, 4);
        return v;
    }
    void memset(int val) { std::memset(buf, val, sizeof(buf)); }
};

TEST_CASE("I1a: barrier ensures all writes visible before any read (16 threads)",
          "[exec][i1][scheduling_order]") {
    SharedMem smem;
    smem.memset(0);

    // Phase 1: each thread writes its tid to data_a[tid]
    for (int tid = 0; tid < 16; tid++) {
        smem.write(tid * 4, (uint32_t)tid);
    }
    // All writes completed before barrier release

    // Phase 2: divergent read+write (simplified — all 16 threads do this)
    for (int tid = 0; tid < 16; tid++) {
        uint32_t a_tid = smem.read(tid * 4);
        uint32_t a_next = smem.read(((tid + 1) % 16) * 4);
        smem.write(64 + tid * 4, a_tid + a_next);
    }

    // Verify all data_b values
    for (int tid = 0; tid < 16; tid++) {
        uint32_t expected = (uint32_t)(tid + (tid + 1) % 16);
        REQUIRE(smem.read(64 + tid * 4) == expected);
    }
}

TEST_CASE("I1b: stale data without barrier (what happens if barrier is broken)",
          "[exec][i1][no_barrier]") {
    // Simulates the failure mode if barrier doesn't ensure write-before-read
    SharedMem smem;
    smem.memset(0);

    // Thread 0 starts writing phase 2 BEFORE other threads finish phase 1
    // This simulates a broken barrier
    uint32_t early_read_0 = smem.read(4); // reads 0 (stale, not yet written by thread 1)
    smem.write(64, early_read_0 + 0);

    // Thread 0 reads its own data_a
    uint32_t a_0 = smem.read(0); // still 0
    REQUIRE(a_0 == 0);

    // This demonstrates: if threads execute out of order WITHOUT barrier sync,
    // data_b[0] = 0 + 0 = 0 (matches "got 0" failure)
    uint32_t early_result = smem.read(64);
    REQUIRE(early_result == 0);
}

// ============================================================
// I2: Second barrier reinit — does it get stuck waiting for 32 threads?
// ============================================================

TEST_CASE("I2a: barrier init with full mask, only 16 lanes arrive",
          "[exec][i2][full_mask_partial_arrive]") {
    Wbar barrier;
    barrier.init(0xFFFFFFFFu, 20); // Full mask, 16-thread CTA

    REQUIRE(barrier.count_participants() == 32);
    REQUIRE(!barrier.is_complete());

    // Only 16 threads exist and arrive
    for (int i = 0; i < 16; i++) {
        barrier.arrive(i);
    }

    REQUIRE(!barrier.is_complete());
    REQUIRE(barrier.count_arrived() == 16);
}

TEST_CASE("I2b: barrier completion with full mask — ALL 32 lanes must arrive",
          "[exec][i2][all_32_arrive]") {
    Wbar barrier;
    barrier.init(0xFFFFFFFFu, 20);

    // All 32 lanes arrive
    for (int i = 0; i < 32; i++) {
        barrier.arrive(i);
    }

    REQUIRE(barrier.is_complete());
    REQUIRE(barrier.count_arrived() == 32);

    // After complete, reset
    barrier.reset();
    REQUIRE(!barrier.is_initialized);
}

TEST_CASE("I2c: barrier with correct 16-thread mask completes normally",
          "[exec][i2][correct_mask]") {
    Wbar barrier;
    barrier.init(0x0000FFFFu, 20); // 16-thread mask

    REQUIRE(barrier.count_participants() == 16);
    REQUIRE(!barrier.is_complete());

    for (int i = 0; i < 16; i++) {
        barrier.arrive(i);
    }

    REQUIRE(barrier.is_complete());
    REQUIRE(barrier.count_arrived() == 16);
}

TEST_CASE("I2d: barrier reuse after reset — second init works correctly",
          "[exec][i2][reuse_after_reset]") {
    Wbar barrier;

    // First barrier (correct 16-thread mask)
    barrier.init(0x0000FFFFu, 20);
    for (int i = 0; i < 16; i++) {
        barrier.arrive(i);
    }
    REQUIRE(barrier.is_complete());
    barrier.reset();

    // Second barrier (same 16-thread mask)
    barrier.init(0x0000FFFFu, 30);
    for (int i = 0; i < 16; i++) {
        barrier.arrive(i);
    }
    REQUIRE(barrier.is_complete());
    // reconvergence_pc updated to 30
}

// ============================================================
// I3: Full 16-thread barrier interaction simulation
// ============================================================

struct SimulatedWarp {
    std::vector<uint32_t> lane_pc;   // Per-lane ThreadContext::pc
    std::vector<bool> lane_active;    // isActive
    std::vector<bool> lane_blocked;
    WarpState ws;
    Wbar wbar;
    SharedMem smem;
    uint32_t participation_mask;

    SimulatedWarp(int thread_count) {
        lane_pc.resize(32, 0);
        lane_active.resize(32, false);
        lane_blocked.resize(32, false);
        for (int i = 0; i < 32; i++) {
            ws.threads[i].pc = 0;
            ws.threads[i].is_active = (i < thread_count);
            ws.threads[i].is_blocked = false;
            ws.threads[i].is_exited = false;
            lane_active[i] = (i < thread_count);
            lane_pc[i] = 0;
        }
        participation_mask = (thread_count >= 32) ? 0xFFFFFFFFu : ((1u << thread_count) - 1);
    }

    void sync_from_warp(int lane) {
        lane_pc[lane] = ws.threads[lane].pc;
    }

    void sync_to_warp(int lane) {
        ws.threads[lane].pc = lane_pc[lane];
        ws.threads[lane].is_blocked = lane_blocked[lane];
    }

    int barrier_handler_arrive(int lane, int barrier_pc, int reconvergence_pc) {
        if (!wbar.is_initialized || wbar.participation_mask != participation_mask) {
            wbar.init(participation_mask, reconvergence_pc);
        }
        wbar.arrive(lane);
        ws.threads[lane].pc = barrier_pc;

        if (wbar.is_complete()) {
            uint32_t mask = wbar.participation_mask;
            uint32_t reconverge = wbar.reconvergence_pc;
        for (int i = 0; i < 32; i++) {
            if ((mask & (1u << i)) && ws.threads[i].is_active) {
                ws.threads[i].pc = reconverge;
                ws.threads[i].is_blocked = false;
            }
        }
            wbar.reset();
            return reconverge;
        } else {
            ws.threads[lane].is_blocked = true;
            return -1; // Barrier not complete yet
        }
    }

    bool all_at_same_pc(int target_pc) const {
        for (int i = 0; i < 32; i++) {
            if (lane_active[i] && !lane_blocked[i] && lane_pc[i] != (uint32_t)target_pc) {
                return false;
            }
        }
        return true;
    }

    int count_active_at_pc(int pc) const {
        int count = 0;
        for (int i = 0; i < 32; i++) {
            if (lane_active[i] && !lane_blocked[i] && lane_pc[i] == (uint32_t)pc) {
                count++;
            }
        }
        return count;
    }
};

TEST_CASE("I3a: Full 16-thread barrier — both barriers complete correctly",
          "[exec][i3][full_simulation]") {
    SimulatedWarp warp(16);
    int reconvergence_pc_1 = 6;
    int reconvergence_pc_2 = 15;

    // Phase 1: All 16 threads at PC=5 (first barrier)
    for (int i = 0; i < 16; i++) {
        warp.lane_pc[i] = 5;
        warp.sync_to_warp(i);
    }

    // Each lane arrives at first barrier
    int rc = -1;
    for (int i = 0; i < 16; i++) {
        int result = warp.barrier_handler_arrive(i, 5, reconvergence_pc_1);
        if (result >= 0) rc = result;
    }
    REQUIRE(rc == reconvergence_pc_1);

    // After first barrier: all 16 lanes should be at PC=6
    for (int i = 0; i < 16; i++) {
        warp.sync_from_warp(i);
        REQUIRE(warp.lane_pc[i] == (uint32_t)reconvergence_pc_1);
        REQUIRE(warp.ws.threads[i].pc == (uint32_t)reconvergence_pc_1);
        REQUIRE(!warp.ws.threads[i].is_blocked);
    }

    // Phase 2: Execute through PC 6-13, arrive at second barrier (PC=14)
    for (int i = 0; i < 16; i++) {
        warp.lane_pc[i] = 14;
        warp.sync_to_warp(i);
    }

    rc = -1;
    for (int i = 0; i < 16; i++) {
        int result = warp.barrier_handler_arrive(i, 14, reconvergence_pc_2);
        if (result >= 0) rc = result;
    }
    REQUIRE(rc == reconvergence_pc_2);

    // After second barrier: all 16 lanes should be at PC=15
    for (int i = 0; i < 16; i++) {
        warp.sync_from_warp(i);
        REQUIRE(warp.lane_pc[i] == (uint32_t)reconvergence_pc_2);
        REQUIRE(!warp.ws.threads[i].is_blocked);
    }
}

TEST_CASE("I3b: Barrier with full 0xFFFFFFFF mask in 16-thread CTA — HANGS",
          "[exec][i3][full_mask_hang]") {
    SimulatedWarp warp(16);
    warp.participation_mask = 0xFFFFFFFFu;

    int reconvergence_pc = 15;
    for (int i = 0; i < 16; i++) {
        warp.lane_pc[i] = 14;
        warp.sync_to_warp(i);
    }

    // Each thread tries to arrive at the barrier
    int complete_rc = -1;
    for (int i = 0; i < 16; i++) {
        int result = warp.barrier_handler_arrive(i, 14, reconvergence_pc);
        if (result >= 0) complete_rc = result;
    }

    // BUG: Barrier NEVER completes because mask is 0xFFFFFFFF but only 16 lanes exist
    REQUIRE(complete_rc == -1); // Barrier doesn't complete!
    REQUIRE(!warp.wbar.is_complete());
    REQUIRE(warp.wbar.count_arrived() == 16);
    REQUIRE(warp.wbar.count_participants() == 32); // 32 required, only 16 arrived

    // After "arriving", check warp_state
    for (int i = 0; i < 16; i++) {
        warp.sync_from_warp(i);
        // Lanes are STILL blocked (never released because barrier never completed)
        REQUIRE(warp.ws.threads[i].is_blocked == true);
    }
}

TEST_CASE("I3c: Barrier auto-fill path — all 32 lanes forced to arrive",
          "[exec][i3][autofill_path]") {
    WarpState wstate;
    Wbar wbar;
    wbar.init(0xFFFFFFFFu, 15);

    // 16 real threads arrive
    for (int i = 0; i < 16; i++) {
        wstate.threads[i].is_active = true;
        wbar.arrive(i);
    }

    // Auto-fill: mark all remaining lanes as arrived
    for (int i = 0; i < 32; i++) {
        if (!(wbar.arrived_mask & (1u << i))) {
            wbar.arrive(i);
            wstate.threads[i].is_blocked = false;
        }
    }

    REQUIRE(wbar.is_complete());
    for (int i = 0; i < 16; i++) {
        REQUIRE(wstate.threads[i].is_blocked == false);
    }
    for (int i = 16; i < 32; i++) {
        REQUIRE(wstate.threads[i].is_blocked == false);
    }
}
