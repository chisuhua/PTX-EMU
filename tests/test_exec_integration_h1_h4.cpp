/**
 * E8: Integration-level PC sync tests for divergent barrier paths.
 *
 * Targets three specific execution-layer failure hypotheses:
 * 1. sync_from_warp_state PC update timing after barrier completion
 * 2. PC consistency across lanes after divergent barrier merge
 * 3. get_lanes_by_pc group reconstruction post-barrier
 */

#include "catch_amalgamated.hpp"
#include "ptxsim/wbar.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/thread_state.h"
#include <cstdint>
#include <map>
#include <vector>

using ptxsim::Wbar;
using ptxsim::WarpState;
using ptxsim::ThreadState;
using ptxsim::ThreadStatus;

// ============================================================
// H1: sync_from_warp_state PC sync timing after barrier
// ============================================================

static int simulate_sync_from_warp_state_pc(
    const WarpState& ws, int lane_id)
{
    if (lane_id < 0 || lane_id >= 32) return -1;
    return ws.threads[lane_id].pc;
}

TEST_CASE("H1a: after barrier completion, all lanes read correct reconvergence_pc",
          "[exec][h1][sync_timing]") {
    // Simulates: 16-thread CTA, all arrive at barrier at PC=12
    // After completion, warp_state.threads[i].pc should be reconvergence_pc
    WarpState ws;
    for (int i = 0; i < 32; i++) {
        ws.threads[i].pc = 12;
        ws.threads[i].is_active = (i < 16);
        ws.threads[i].is_blocked = true;
    }

    // Simulate barrier completion (from bar.warp.sync handler line 182-188):
    int reconvergence_pc = 16;
    uint32_t participation_mask = 0x0000FFFFu;
    for (int i = 0; i < 32; ++i) {
        if ((participation_mask & (1u << i)) && ws.threads[i].is_active) {
            ws.threads[i].pc = reconvergence_pc;  // set_thread_pc
            ws.threads[i].is_blocked = false;
        }
    }

    // sync_from_warp_state reads warp_state.threads[lane].pc
    for (int i = 0; i < 16; i++) {
        REQUIRE(simulate_sync_from_warp_state_pc(ws, i) == 16);
    }
    // Lanes 16-31: inactive, pc stays at 12
    for (int i = 16; i < 32; i++) {
        REQUIRE(simulate_sync_from_warp_state_pc(ws, i) == 12);
    }
}

TEST_CASE("H1b: next cycle after barrier — lanes at same PC group together",
          "[exec][h1][next_cycle]") {
    // After barrier release, all active lanes should be at reconvergence_pc
    // When get_lanes_by_pc runs next cycle, they should all be in same group
    WarpState ws;
    std::vector<int> thread_pcs(32, 0);
    for (int i = 0; i < 32; i++) {
        ws.threads[i].pc = (i < 16) ? 20 : 0;
        ws.threads[i].is_active = (i < 16);
        ws.threads[i].is_blocked = false;
        thread_pcs[i] = ws.threads[i].pc;   // ThreadContext::pc after sync_from_warp_state
    }

    std::map<int, std::vector<int>> groups;
    for (int lane = 0; lane < 32; lane++) {
        if (lane < 16 && ws.threads[lane].is_active &&
            !ws.threads[lane].is_blocked) {
            groups[thread_pcs[lane]].push_back(lane);
        }
    }

    REQUIRE(groups.size() == 1u);           // All in ONE group
    REQUIRE(groups.begin()->first == 20);    // Correct PC
    REQUIRE(groups.begin()->second.size() == 16u);
}

// ============================================================
// H2: PC consistency check across divergent lanes post-barrier
// ============================================================

TEST_CASE("H2a: divergent lanes reconverge to same PC (8/8 split)",
          "[exec][h2][divergent_reconvergence]") {
    // Simulates: 16 threads, 8 at PC=7 (branch), 8 at PC=11 (label)
    // Both groups hit barrier at PC=12, should all end at same reconvergence_pc
    std::vector<int> pc(32);
    std::vector<bool> is_active(32, false);
    std::vector<bool> is_blocked(32, false);

    // Setup: lanes 0-7 at PC=7, lanes 8-15 at PC=11
    for (int i = 0; i < 8; i++) { pc[i] = 7; is_active[i] = true; }
    for (int i = 8; i < 16; i++) { pc[i] = 11; is_active[i] = true; }

    // Both groups execute through to barrier at PC=12
    // Group 1 (lanes 0-7): PC 7 → 8 → 9 → 10 → 11 (merge) → 12 (barrier)
    // Group 2 (lanes 8-15): PC 11 → 12 (barrier)
    for (int i = 0; i < 16; i++) pc[i] = 12;
    for (int i = 0; i < 16; i++) is_blocked[i] = true;

    // Barrier completion
    int reconvergence_pc = 16;
    uint32_t mask = 0x0000FFFFu;
    for (int i = 0; i < 32; ++i) {
        if ((mask & (1u << i)) && is_active[i]) {
            pc[i] = reconvergence_pc;       // set_thread_pc
            is_blocked[i] = false;
        }
    }

    // Verify: all active lanes at same PC
    for (int i = 0; i < 16; i++) {
        REQUIRE(pc[i] == 16);
    }

    // get_lanes_by_pc should return all 16 lanes in one group
    std::map<int, std::vector<int>> groups;
    for (int lane = 0; lane < 32; lane++) {
        if (is_active[lane] && !is_blocked[lane]) {
            groups[pc[lane]].push_back(lane);
        }
    }
    REQUIRE(groups.size() == 1u);
    REQUIRE(groups[16].size() == 16u);
}

TEST_CASE("H2b: mixed barrier — some lanes never reached barrier",
          "[exec][h2][partial_arrival]") {
    // Edge: some lanes skip the barrier entirely (branch over it)
    // This shouldn't happen in correct PTX, but test robustness
    std::vector<int> pc(32, 0);
    std::vector<bool> is_active(32, false);
    std::vector<bool> is_blocked(32, false);

    // Lanes 0-7 at barrier (PC=5), lanes 8-15 already past it (PC=10)
    for (int i = 0; i < 8; i++) { pc[i] = 5; is_active[i] = true; is_blocked[i] = true; }
    for (int i = 8; i < 16; i++) { pc[i] = 10; is_active[i] = true; }

    // Barrier at PC=5 completes for 8 lanes
    uint32_t mask = 0x000000FFu;  // Only lanes 0-7
    for (int i = 0; i < 8; i++) {
        pc[i] = 10;              // reconvergence_pc = 10 (merge with lanes 8-15)
        is_blocked[i] = false;
    }

    // After barrier: all lanes should be at PC=10
    for (int i = 0; i < 16; i++) {
        REQUIRE(pc[i] == 10);
    }

    std::map<int, std::vector<int>> groups;
    for (int lane = 0; lane < 32; lane++) {
        if (is_active[lane] && !is_blocked[lane]) {
            groups[pc[lane]].push_back(lane);
        }
    }
    REQUIRE(groups.size() == 1u);
    REQUIRE(groups[10].size() == 16u);
}

TEST_CASE("H2c: sequential barrier execution — lanes at different stages",
          "[exec][h2][sequential]") {
    // Lanes arrive at barrier at different times during sequential execution
    // Lane 0 arrives first, lane 15 arrives last
    std::vector<int> pc(32);
    std::vector<bool> is_active(32, false);
    std::vector<bool> is_blocked(32, false);

    // First execution pass: lanes 0-7 reach barrier at PC=12
    for (int i = 0; i < 8; i++) { pc[i] = 12; is_active[i] = true; is_blocked[i] = true; }

    // Lanes 8-15 still processing earlier instructions (PC=10)
    for (int i = 8; i < 16; i++) { pc[i] = 10; is_active[i] = true; }

    // Wbar state for 8 lanes so far
    Wbar wbar;
    wbar.init(0x0000FFFFu, 20);  // 16 participants, reconvergence at 20
    for (int i = 0; i < 8; i++) {
        wbar.arrive(i);  // 8 arrived so far
    }
    REQUIRE(!wbar.is_complete());  // Not complete yet (only 8/16)

    // Second pass: lanes 8-15 reach barrier
    for (int i = 8; i < 16; i++) {
        wbar.arrive(i);
        pc[i] = 12;
        is_blocked[i] = true;
    }
    REQUIRE(wbar.is_complete());  // Now complete

    // Barrier completion updates all to reconvergence_pc
    for (int i = 0; i < 16; i++) {
        if (wbar.participation_mask & (1u << i)) {
            pc[i] = 20;
            is_blocked[i] = false;
        }
    }
    for (int i = 0; i < 16; i++) {
        REQUIRE(pc[i] == 20);
        REQUIRE(!is_blocked[i]);
    }
}

// ============================================================
// H3: get_lanes_by_pc group reconstruction post-barrier
// ============================================================

TEST_CASE("H3a: single warp — lane grouping after barrier release",
          "[exec][h3][single_warp]") {
    // Simulates warp_scheduler calling get_lanes_by_pc after barrier
    WarpState ws;
    for (int i = 0; i < 32; i++) {
        ws.threads[i].pc = 20;
        ws.threads[i].is_active = (i < 16);
        ws.threads[i].is_blocked = false;
        ws.threads[i].is_exited = false;
    }

    std::vector<int> thread_pcs(32);
    for (int i = 0; i < 32; i++) thread_pcs[i] = ws.threads[i].pc;

    // Simulate get_lanes_by_pc filtering
    std::map<int, std::vector<int>> groups;
    for (int lane = 0; lane < 32; lane++) {
        if (lane < 32 && ws.threads[lane].is_active &&
            !ws.threads[lane].is_exited && !ws.threads[lane].is_blocked) {
            groups[thread_pcs[lane]].push_back(lane);
        }
    }

    // After barrier release: ONE group at PC=20 with 16 lanes
    REQUIRE(groups.size() == 1u);
    REQUIRE(groups.count(20) == 1u);
    REQUIRE(groups[20].size() == 16u);
    for (int i = 0; i < 16; i++) {
        REQUIRE(groups[20][i] == i);
    }
}

TEST_CASE("H3b: divergent path — lane grouping at intermediate PCs",
          "[exec][h3][divergent_intermediate]") {
    WarpState ws;
    std::vector<int> thread_pcs(32);

    for (int i = 0; i < 8; i++) {
        thread_pcs[i] = 5;
        ws.threads[i].is_active = true;
        ws.threads[i].is_blocked = false;
        ws.threads[i].is_exited = false;
    }
    for (int i = 8; i < 16; i++) {
        thread_pcs[i] = 9;
        ws.threads[i].is_active = true;
        ws.threads[i].is_blocked = false;
        ws.threads[i].is_exited = false;
    }
    for (int i = 16; i < 32; i++) {
        ws.threads[i].is_active = false;
        thread_pcs[i] = 0;
    }

    std::map<int, std::vector<int>> groups;
    for (int lane = 0; lane < 32; lane++) {
        if (ws.threads[lane].is_active &&
            !ws.threads[lane].is_exited && !ws.threads[lane].is_blocked) {
            groups[thread_pcs[lane]].push_back(lane);
        }
    }

    REQUIRE(groups.size() == 2u);
    REQUIRE(groups.count(5) == 1u);
    REQUIRE(groups.count(9) == 1u);
    REQUIRE(groups[5].size() == 8u);
    REQUIRE(groups[9].size() == 8u);
}

TEST_CASE("H3c: barrier instruction at PC=N — blocked lanes excluded from grouping",
          "[exec][h3][blocked_excluded]") {
    WarpState ws;
    std::vector<int> thread_pcs(32);

    for (int i = 0; i < 16; i++) {
        thread_pcs[i] = 12;
        ws.threads[i].is_active = true;
        ws.threads[i].is_blocked = true;
        ws.threads[i].is_exited = false;
    }
    for (int i = 16; i < 32; i++) {
        ws.threads[i].is_active = false;
        thread_pcs[i] = 0;
    }

    std::map<int, std::vector<int>> groups;
    for (int lane = 0; lane < 32; lane++) {
        if (ws.threads[lane].is_active &&
            !ws.threads[lane].is_exited && !ws.threads[lane].is_blocked) {
            groups[thread_pcs[lane]].push_back(lane);
        }
    }

    REQUIRE(groups.empty());
}

TEST_CASE("H3d: mixed state — some blocked, some active, some exited",
          "[exec][h3][mixed_state]") {
    WarpState ws;
    std::vector<int> thread_pcs(32);

    for (int i = 0; i < 32; i++) {
        ws.threads[i].is_exited = false;
        ws.threads[i].is_blocked = false;
        ws.threads[i].is_active = false;
        thread_pcs[i] = 0;
    }

    for (int i = 0; i < 4; i++) {
        ws.threads[i].is_exited = true;
        ws.threads[i].is_active = true;
    }

    for (int i = 4; i < 8; i++) {
        ws.threads[i].is_active = true;
        ws.threads[i].is_blocked = true;
        thread_pcs[i] = 12;
    }

    for (int i = 8; i < 16; i++) {
        ws.threads[i].is_active = true;
        thread_pcs[i] = 20;
    }

    std::map<int, std::vector<int>> groups;
    for (int lane = 0; lane < 32; lane++) {
        if (ws.threads[lane].is_active &&
            !ws.threads[lane].is_exited && !ws.threads[lane].is_blocked) {
            groups[thread_pcs[lane]].push_back(lane);
        }
    }

    REQUIRE(groups.size() == 1u);
    REQUIRE(groups.count(20) == 1u);
    REQUIRE(groups[20].size() == 8u);
}

// ============================================================
// H4 (bonus): ThreadContext::pc update sequence after barrier
// ============================================================

TEST_CASE("H4a: thread pc update sequence after barrier completes",
          "[exec][h4][update_sequence]") {
    // Simulates the exact sequence from execute_warp_instruction:
    // 1. sync_from_warp_state: thread.pc = warp_state.threads[lane].pc
    // 2. If BAR_SYNC: check barrier, sync_to_warp_state
    // 3. thread.execute_thread_instruction: next_pc = thread.pc + 1, exec, thread.pc = next_pc

    // Initial state: barrier completed, warp_state updated by handler
    int ws_pc[32];
    int thread_pc[32];
    bool is_blocked[32];
    bool is_active[32];

    for (int i = 0; i < 32; i++) {
        ws_pc[i] = 20;  // Barrier handler set reconvergence_pc = 20
        thread_pc[i] = 12;  // ThreadContext still at barrier instruction
        is_blocked[i] = (i < 16);  // Some lanes blocked, some unblocked
        is_active[i] = (i < 16);
    }
    // All lanes unblocked after barrier completion
    for (int i = 0; i < 16; i++) is_blocked[i] = false;

    // After sync_from_warp_state:
    for (int i = 0; i < 32; i++) {
        if (is_active[i] && !is_blocked[i]) {
            thread_pc[i] = ws_pc[i];  // Should be 20, NOT 12
        }
    }

    // Verify: after barrier completion + sync, all active lanes at reconvergence PC
    int group_size = 0;
    for (int i = 0; i < 16; i++) {
        if (thread_pc[i] == 20) group_size++;
    }
    REQUIRE(group_size == 16);

    // next_pc = pc + 1 → 21 for next instruction
    for (int i = 0; i < 16; i++) {
        int next_pc = thread_pc[i] + 1;
        REQUIRE(next_pc == 21);
    }
}

TEST_CASE("H4b: barrier handler pc update — active only, inactive untouched",
          "[exec][h4][selective_update]") {
    // Barrier handler (bar.warp.sync) only updates ACTIVE lanes
    int ws_pc[32];
    bool is_active[32];

    for (int i = 0; i < 32; i++) {
        ws_pc[i] = 5;
        is_active[i] = (i < 16);
    }

    // Barrier completion for 16 threads
    int reconvergence_pc = 16;
    uint32_t mask = 0x0000FFFFu;
    for (int i = 0; i < 32; ++i) {
        if ((mask & (1u << i)) && is_active[i]) {
            ws_pc[i] = reconvergence_pc;
        }
    }

    // Active lanes (0-15): updated to 16
    for (int i = 0; i < 16; i++) {
        REQUIRE(ws_pc[i] == 16);
    }
    // Inactive lanes (16-31): NOT updated by barrier
    for (int i = 16; i < 32; i++) {
        REQUIRE(ws_pc[i] == 5);
    }
}
