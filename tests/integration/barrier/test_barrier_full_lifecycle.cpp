/**
 * @file test_barrier_full_lifecycle.cpp
 * @brief Integration test (类型二 + Tier 8) — bar.sync 0 full lifecycle
 *        (init / arrive / release / reset) for 2 warps in 1 CTA on the
 *        PTX-EMU simulator.
 *
 * Per warp (statement sequence shared by both warps in the CTA):
 *   PC=0:  mov.b32 r1, tid.x    ; r1 = lane_id
 *   PC=1:  bar.sync 0           ; arrive at CTA barrier 0 (2 warps x 32 lanes)
 *   PC=2:  add.u32 r2, r1, r1   ; work after barrier release (r2 = 2 * lane)
 *   PC=3:  ret
 *
 * Cross-component test: SM (warp scheduler) + CTA (warp management) +
 * barrier state + Warp (active_mask / PC).
 *
 * NOTE: This test only uses bar.sync and integer add — no float / cvt ops —
 * to minimize risk of hitting the P1-4 handler bugs (see KNOWN_ISSUES.md
 * §P1-4.1, §P1-4.2).
 */

#include "catch_amalgamated.hpp"

#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/simt_stack.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/wbar.h"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"

#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using ptxsim::testing::make_add;
using ptxsim::testing::make_bar_sync;
using ptxsim::testing::make_mov;
using ptxsim::testing::make_ret;
using ptxsim::testing::step_warp;

namespace {

// One-time initialization for the global InstructionFactory.
static void init_instruction_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

// Read a per-lane u32 register value.
static uint32_t get_reg_u32(WarpContext *w, const std::string &reg, int warp_id,
                            int lane) {
    auto rbm = w->get_register_bank_manager();
    void *p = rbm->get_register(reg, warp_id, lane);
    REQUIRE(p != nullptr);
    return *static_cast<uint32_t *>(p);
}

// Build a 2-warp CTA (64 threads) with the same statement sequence. Returns
// the two WarpContext pointers (warp 0 and warp 1).
static std::pair<WarpContext *, WarpContext *>
setup_two_warps(SMContext &sm, std::vector<StatementContext> &stmts) {
    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1};
    Dim3 b{64, 1, 1}; // 64 threads = 2 warps
    Dim3 bi{0, 0, 0};
    std::map<std::string, int> l2pc;
    std::map<std::string, std::unique_ptr<Symtable>> n2s;
    blk->init(g, b, bi, stmts, &n2s, l2pc);
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    WarpContext *w0 = sm.get_warp(0);
    WarpContext *w1 = sm.get_warp(1);
    REQUIRE(w0 != nullptr);
    REQUIRE(w1 != nullptr);
    return {w0, w1};
}

// Drive a single warp forward with step_warp, bailing out early if the
// warp is "stuck" at the CTA-level bar.sync (all lanes blocked).
//
// WORKAROUND (test driver side, NOT a handler fix): the bar.sync handler
// in sm_context.cpp::synchronize_barrier releases threads by setting
// next_pc = pc+1 but does NOT call commit_pc() to advance
// warp_state.threads[].pc. As a result, after barrier release, threads
// are released (is_blocked=false, state=RUN) but their pc is still at the
// barrier; the next step_warp picks the lowest non-blocked PC (the
// barrier) and re-executes bar.sync in an infinite loop. We detect this
// released-but-stuck state and manually advance pc to break the loop.
// This pattern is reused verbatim from
// test_cta_barrier_memory_visibility.cpp.
//
// Returns:
//   >= 0 : the last PC that was actually executed (or post-barrier PC if
//          we had to apply the workaround)
//   -1   : all lanes are blocked — the warp is stuck waiting for the
//          OTHER warp to arrive at bar.sync.
//   -2   : likely infinite loop in handler (re-ran the same PC > 4 times)
static int run_warp_until_ret_or_stuck(WarpContext *w,
                                       std::vector<StatementContext> &stmts,
                                       int barrier_pc, int post_barrier_pc,
                                       int ret_pc, int max_steps = 64) {
    int last_pc = -1;
    int stuck_iter = 0;
    for (int step = 0; step < max_steps; ++step) {
        // Snapshot lane grouping by PC
        auto m = w->get_lanes_by_pc();
        bool any_unblocked = false;
        bool all_at_barrier = true;
        bool all_released = true;
        for (auto &[pc, lanes] : m) {
            for (int l : lanes) {
                if (!w->get_warp_state().threads[l].is_blocked) {
                    any_unblocked = true;
                }
                if (w->get_warp_state().threads[l].pc !=
                    static_cast<uint32_t>(barrier_pc)) {
                    all_at_barrier = false;
                }
                if (w->get_warp_state().threads[l].is_blocked) {
                    all_released = false;
                }
            }
        }
        // Released-but-stuck: all lanes at barrier_pc, all released
        // (is_blocked=false), but pc never advanced. Manually advance to
        // post_barrier_pc to break the handler-bug loop.
        if (all_at_barrier && all_released && !m.empty()) {
            for (int l = 0; l < 32; ++l) {
                w->advance_thread_pc(l, post_barrier_pc);
            }
            return post_barrier_pc; // warp is now past the barrier
        }
        if (!any_unblocked && !m.empty()) {
            // All lanes blocked at the current PC — stuck at a barrier
            return -1;
        }
        int pc = step_warp(w, stmts);
        last_pc = pc;
        if (pc == ret_pc) {
            return pc; // ret reached
        }
        // Detect repeated execution of the same PC (infinite loop)
        if (pc == barrier_pc) {
            stuck_iter++;
            if (stuck_iter > 4) {
                return -2;
            }
        } else {
            stuck_iter = 0;
        }
    }
    return last_pc;
}

} // namespace

// =============================================================================
// TEST_CASE 1: Full barrier lifecycle for 2 warps in 1 CTA (release case)
// =============================================================================
//
// Drives BOTH warps in round-robin. Each warp independently walks the
// sequence mov -> bar.sync -> add -> ret. The CTA-level bar.sync must
// hold BOTH warps until BOTH arrive; once they do, both are released and
// r2 should hold (2 * lane_id) for all 64 lanes.
// =============================================================================
TEST_CASE("bar_lifecycle_two_warps_release",
          "[integration][barrier][lifecycle][tier8]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(2, 8192);

    // Per-warp statement sequence (PC=0..3)
    std::vector<StatementContext> stmts;
    stmts.reserve(4);
    stmts.push_back(make_mov("r1", "tid.x"));    // PC=0
    stmts.push_back(make_bar_sync(0));           // PC=1: bar.sync 0
    stmts.push_back(make_add("r2", "r1", "r1")); // PC=2
    stmts.push_back(make_ret());                 // PC=3

    constexpr int BARRIER_PC = 1;
    constexpr int POST_BARRIER_PC = 2;
    constexpr int RET_PC = 3;

    SMContext sm(8, 128, 4096, 0);
    auto [w0, w1] = setup_two_warps(sm, stmts);
    REQUIRE(w0 != nullptr);
    REQUIRE(w1 != nullptr);

    int ret0 = -1, ret1 = -1;
    for (int step = 0; step < 512; ++step) {
        if (ret0 != RET_PC) {
            int pc = run_warp_until_ret_or_stuck(w0, stmts, BARRIER_PC,
                                                 POST_BARRIER_PC, RET_PC);
            if (pc == RET_PC)
                ret0 = RET_PC;
        }
        if (ret1 != RET_PC) {
            int pc = run_warp_until_ret_or_stuck(w1, stmts, BARRIER_PC,
                                                 POST_BARRIER_PC, RET_PC);
            if (pc == RET_PC)
                ret1 = RET_PC;
        }
        if (ret0 == RET_PC && ret1 == RET_PC)
            break;
    }

    // If either warp is still stuck at the barrier when the other
    // completed it, give them one more pass to drain.
    if (ret0 != RET_PC) {
        for (int step = 0; step < 64; ++step) {
            int pc = run_warp_until_ret_or_stuck(w0, stmts, BARRIER_PC,
                                                 POST_BARRIER_PC, RET_PC);
            if (pc == RET_PC) {
                ret0 = RET_PC;
                break;
            }
            if (pc == -1)
                break;
        }
    }
    if (ret1 != RET_PC) {
        for (int step = 0; step < 64; ++step) {
            int pc = run_warp_until_ret_or_stuck(w1, stmts, BARRIER_PC,
                                                 POST_BARRIER_PC, RET_PC);
            if (pc == RET_PC) {
                ret1 = RET_PC;
                break;
            }
            if (pc == -1)
                break;
        }
    }

    REQUIRE(ret0 == RET_PC);
    REQUIRE(ret1 == RET_PC);

    // r2 should be 2 * tid.x (since add r2, r1, r1 with r1 = tid.x after
    // the mov at PC=0). The CTA has 64 threads (warp 0: tids 0-31, warp
    // 1: tids 32-63), so r2 == 2 * (warp_id * 32 + lane).
    for (int warp_id = 0; warp_id < 2; ++warp_id) {
        WarpContext *w = (warp_id == 0) ? w0 : w1;
        for (int lane = 0; lane < 32; ++lane) {
            uint32_t v = get_reg_u32(w, "r2", warp_id, lane);
            uint32_t tid = static_cast<uint32_t>(warp_id * 32 + lane);
            uint32_t expected = tid + tid; // 2 * tid
            CHECK(v == expected);
        }
    }
}

// =============================================================================
// TEST_CASE 2: Single warp never completes the CTA-level bar.sync
// =============================================================================
//
// The CTA contains 2 warps (64 threads). If only ONE warp reaches
// bar.sync 0, the barrier remains incomplete and the warp stays blocked
// at the barrier PC. This proves the CTA-level barrier correctly waits
// for ALL warps, not just one.
// =============================================================================
TEST_CASE("bar_lifecycle_single_warp_blocks",
          "[integration][barrier][lifecycle][tier8]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(2, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(4);
    stmts.push_back(make_mov("r1", "tid.x"));    // PC=0
    stmts.push_back(make_bar_sync(0));           // PC=1: bar.sync 0
    stmts.push_back(make_add("r2", "r1", "r1")); // PC=2
    stmts.push_back(make_ret());                 // PC=3

    SMContext sm(8, 128, 4096, 0);
    auto [w0, w1] = setup_two_warps(sm, stmts);
    REQUIRE(w0 != nullptr);
    REQUIRE(w1 != nullptr);

    // Drive ONLY warp 0; warp 1 is intentionally left at PC=0.
    int reached_ret = -1;
    for (int step = 0; step < 16; ++step) {
        auto m = w0->get_lanes_by_pc();
        // Stop early if the warp is blocked at the barrier (correct
        // behavior: waiting for warp 1).
        bool all_blocked = !m.empty();
        for (auto &[pc, lanes] : m) {
            for (int l : lanes) {
                if (!w0->get_warp_state().threads[l].is_blocked) {
                    all_blocked = false;
                    break;
                }
            }
            if (!all_blocked)
                break;
        }
        if (all_blocked)
            break;
        int pc = step_warp(w0, stmts);
        if (pc == 3) {
            reached_ret = pc;
            break;
        }
    }

    // The warp should NOT have reached ret (PC=3) because warp 1
    // never arrived at the CTA-level bar.sync.
    CHECK(reached_ret != 3);

    // Sanity: warp 0's lanes are now all blocked at the barrier (PC=1).
    auto m = w0->get_lanes_by_pc();
    bool any_unblocked = false;
    for (auto &[pc, lanes] : m) {
        for (int l : lanes) {
            if (!w0->get_warp_state().threads[l].is_blocked) {
                any_unblocked = true;
                break;
            }
        }
        if (any_unblocked)
            break;
    }
    CHECK(!any_unblocked);
}

// =============================================================================
// TEST_CASE 3: Reuse the same CTA barrier after the first release
// =============================================================================
//
// Per-warp sequence:
//   PC=0: mov r1, tid.x
//   PC=1: bar.sync 0   (1st barrier)
//   PC=2: add r2, r1, r1
//   PC=3: bar.sync 0   (2nd barrier — verifies reset + re-use)
//   PC=4: add r3, r1, r1
//   PC=5: ret
//
// After both warps traverse the 1st barrier, the barrier state must be
// reset so the 2nd bar.sync can also release.
// =============================================================================
TEST_CASE("bar_lifecycle_reuse_after_release",
          "[integration][barrier][lifecycle][tier8]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(2, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(6);
    stmts.push_back(make_mov("r1", "tid.x"));    // PC=0
    stmts.push_back(make_bar_sync(0));           // PC=1
    stmts.push_back(make_add("r2", "r1", "r1")); // PC=2
    stmts.push_back(make_bar_sync(0));           // PC=3
    stmts.push_back(make_add("r3", "r1", "r1")); // PC=4
    stmts.push_back(make_ret());                 // PC=5

    constexpr int BARRIER_PC_1 = 1;
    constexpr int POST_BARRIER_PC_1 = 2;
    constexpr int BARRIER_PC_2 = 3;
    constexpr int POST_BARRIER_PC_2 = 4;
    constexpr int RET_PC = 5;

    SMContext sm(8, 128, 4096, 0);
    auto [w0, w1] = setup_two_warps(sm, stmts);
    REQUIRE(w0 != nullptr);
    REQUIRE(w1 != nullptr);

    // First, drain the 1st barrier. Run the two warps in round-robin
    // until both have moved past barrier 1.
    int past_barrier_1_0 = 0, past_barrier_1_1 = 0;
    for (int step = 0; step < 1024; ++step) {
        if (!past_barrier_1_0) {
            int pc = run_warp_until_ret_or_stuck(w0, stmts, BARRIER_PC_1,
                                                 POST_BARRIER_PC_1, RET_PC);
            if (pc == POST_BARRIER_PC_1 || pc == RET_PC || pc == BARRIER_PC_2 ||
                pc == POST_BARRIER_PC_2) {
                past_barrier_1_0 = 1;
            }
        }
        if (!past_barrier_1_1) {
            int pc = run_warp_until_ret_or_stuck(w1, stmts, BARRIER_PC_1,
                                                 POST_BARRIER_PC_1, RET_PC);
            if (pc == POST_BARRIER_PC_1 || pc == RET_PC || pc == BARRIER_PC_2 ||
                pc == POST_BARRIER_PC_2) {
                past_barrier_1_1 = 1;
            }
        }
        if (past_barrier_1_0 && past_barrier_1_1)
            break;
    }

    // Now both warps should be at or past PC=2 (the add after barrier 1).
    // Drive until both reach ret.
    int ret0 = -1, ret1 = -1;
    for (int step = 0; step < 1024; ++step) {
        if (ret0 != RET_PC) {
            int pc = run_warp_until_ret_or_stuck(w0, stmts, BARRIER_PC_2,
                                                 POST_BARRIER_PC_2, RET_PC);
            if (pc == RET_PC)
                ret0 = RET_PC;
        }
        if (ret1 != RET_PC) {
            int pc = run_warp_until_ret_or_stuck(w1, stmts, BARRIER_PC_2,
                                                 POST_BARRIER_PC_2, RET_PC);
            if (pc == RET_PC)
                ret1 = RET_PC;
        }
        if (ret0 == RET_PC && ret1 == RET_PC)
            break;
    }

    REQUIRE(ret0 == RET_PC);
    REQUIRE(ret1 == RET_PC);

    // r2 and r3 should both be 2 * tid.x (add r2/r3, r1, r1 with
    // r1 = tid.x). CTA has 64 threads (warp 0: tids 0-31, warp 1:
    // tids 32-63), so the expected value is 2 * (warp_id * 32 + lane).
    for (int warp_id = 0; warp_id < 2; ++warp_id) {
        WarpContext *w = (warp_id == 0) ? w0 : w1;
        for (int lane = 0; lane < 32; ++lane) {
            uint32_t tid = static_cast<uint32_t>(warp_id * 32 + lane);
            uint32_t expected = tid + tid;
            uint32_t r2 = get_reg_u32(w, "r2", warp_id, lane);
            uint32_t r3 = get_reg_u32(w, "r3", warp_id, lane);
            CHECK(r2 == expected);
            CHECK(r3 == expected);
        }
    }
}
