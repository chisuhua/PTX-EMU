/**
 * @file test_simt_stack_stale_entry_blocks_lane0.cpp
 * @brief Unit test for the SIMT-stack-driven dispatch gate
 * (BUG-DISPATCH-GATE-LANE0-SKIP hypothesis: residual reduction-loop entries
 *  block lane 0's st.shared at reconvergence_pc).
 *
 * Hypothesis (from cute_rmsnorm debug 2026-06-16):
 *   When a divergent branch is pushed onto the SIMT stack with
 *   `reconvergence_pc = X` and `active_mask` includes lanes still on a
 *   divergent path, the dispatcher MUST block the lanes that have already
 *   arrived at PC=X. This is the **intended** behavior — without it the
 *   divergent branch would lose its non-taken lanes' state.
 *
 *   But: if the entry is **stale** (the reconvergence already happened and
 *   the entry was supposed to be popped by `check_reconvergence`), the
 *   gate will incorrectly block the next instruction that happens to be
 *   at the stale reconvergence_pc.
 *
 *   cute_rmsnorm's 5-iteration reduction loop pushes 5 @%p10 back-edge
 *   entries with `reconvergence_pc = 133` (loop exit). After the loop,
 *   check_reconvergence should pop them all. If one stays, and the
 *   lane 0 st.shared at PC=101 is mistakenly classified as a
 *   "reconvergence point" by the gate, lane 0 is blocked and the
 *   st.shared never executes.
 *
 * Test strategy:
 *   1. Manually push a stale entry with `reconvergence_pc = X` onto
 *      a warp's SIMT stack.
 *   2. Set up two lanes at different PCs: lane 0 at PC=X, lane 1 at PC=X+1.
 *   3. Call `execute_warp_instruction(stmt_at_X, X)` directly.
 *   4. Assert: lane 0 IS executed (st.shared runs) — the entry is stale
 *      and should NOT block the instruction. (This test passes only if
 *      the gate correctly identifies that all `active_mask` lanes are at
 *      `reconvergence_pc=X` when the entry was stale, OR if the entry
 *      has been removed.)
 *
 *   Alternative assertion (if the gate IS the bug): the test should
 *   fail with lane 0 is_blocked=true after the call. That failure
 *   reproduces the cute_rmsnorm bug.
 *
 * NOTE: This test is INVERTED compared to a typical "lock in correct
 * behavior" test. The author wants to **observe** what happens when
 * a stale entry is present. If the test fails, the bug is reproduced
 * and the fix should be: (a) pop stale entries aggressively, or
 * (b) make the gate ignore entries whose `active_mask` lanes have
 * all passed `reconvergence_pc`.
 */

#include "catch_amalgamated.hpp"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/execution_trace.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/predicates.h"
#include "ptxsim/testing/memory_test_utils.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"
#include "ptx_ir/operand_context.h"
#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <cstdint>

using namespace ptxir::factory;
using ptxsim::testing::step_warp;
using ptxsim::testing::setup_pred;
using ptxsim::testing::make_nop;
using ptxsim::testing::make_st_shared_addr;
using ptxsim::testing::make_ret;
using ptxsim::ExecutionTracer;

// ============================================================================
// U-1: A stale SIMTStackEntry with reconvergence_pc=X does NOT block
// lane 0's st.shared at PC=X when all active_mask lanes are at X+1 (past it).
//
// This is the SCENARIO that would occur in cute_rmsnorm if a back-edge
// @%p10 entry with reconv=133 (loop exit) was left on the stack when
// lane 0's st.shared at PC=101 was dispatched. The 101 != 133 check in
// the gate should mean the entry doesn't block — but we want to verify
// the gate only checks `reconvergence_pc` of the TOP entry, not deeper.
// ============================================================================
TEST_CASE("U-1: stale SIMT entry with reconv=X does not block PC=X+1",
          "[barrier][dispatch][simt][regression][BUG-DISPATCH-GATE-LANE0-SKIP]")
{
    ptxsim::testing::init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    // Minimal instruction sequence: just one st.shared for lane 0
    constexpr int PC_ST_SHARED = 0;
    constexpr int PC_RET       = 1;
    constexpr int NUM_STMTS    = 2;

    std::vector<StatementContext> v(NUM_STMTS);
    for (auto& s : v) s = make_nop();

    // PC=0: st.shared.b32 [sdata+0], r_val (lane 0 write)
    v[PC_ST_SHARED] = ptxsim::testing::make_st_shared_addr(
        "sdata", "r_tid", "r_val", Qualifier::Q_B32);
    v[PC_RET] = make_ret();

    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1}, b{32, 1, 1}, bi{0, 0, 0};
    std::map<std::string, int> l2pc = {};
    std::map<std::string, std::unique_ptr<Symtable>> n2s;
    blk->init(g, b, bi, v, &n2s, l2pc);
    blk->sharedMemBytes = 1024;

    SMContext sm(4, 128, 4096, 0);
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    WarpContext* w = sm.get_warp(0);
    REQUIRE(w != nullptr);

    // Manually set up the warp state: lane 0 at PC=0, lanes 1-31 at PC=1
    auto& ws = w->get_warp_state();
    ws.threads[0].pc = PC_ST_SHARED;
    ws.threads[0].next_pc = PC_ST_SHARED;
    ws.threads[0].is_active = true;
    ws.threads[0].is_blocked = false;
    ws.threads[0].is_exited = false;
    ws.threads[0].status = ptxsim::ThreadStatus::Active;
    for (int i = 1; i < 32; i++) {
        ws.threads[i].pc = PC_RET;
        ws.threads[i].next_pc = PC_RET;
        ws.threads[i].is_active = true;
        ws.threads[i].is_blocked = false;
        ws.threads[i].is_exited = false;
        ws.threads[i].status = ptxsim::ThreadStatus::Active;
    }
    w->update_active_mask();

    // *** Inject a STALE SIMTStackEntry with reconvergence_pc=PC_ST_SHARED ***
    // This simulates the cute_rmsnorm scenario where a back-edge entry
    // (reconv=loop_exit) was never popped, and the dispatcher incorrectly
    // classifies the lane 0 st.shared PC as a reconvergence point.
    ptxsim::SIMTStackEntry stale_entry;
    stale_entry.branch_pc = 999;  // some old branch
    stale_entry.reconvergence_pc = PC_ST_SHARED;  // KEY: matches the st.shared PC
    stale_entry.active_mask = 0xFFFFFFFEu;  // lanes 1-31 (already past, at PC=1)
    stale_entry.return_mask = 0xFFFFFFFEu;
    stale_entry.return_pc = PC_ST_SHARED;
    w->get_simt_stack().push(stale_entry);

    // Verify precondition: lane 0 is at PC_ST_SHARED, not blocked
    REQUIRE(ws.threads[0].pc == PC_ST_SHARED);
    REQUIRE(!ws.threads[0].is_blocked);
    REQUIRE(w->is_lane_active(0));

    // Enable tracer to record what actually executes
    ptxsim::ExecutionTracer::enable();
    ptxsim::ExecutionTracer::reset();

    // Drive the dispatch manually (1 step)
    int picked_pc = step_warp(w, v);

    ptxsim::ExecutionTracer::disable();
    const auto& trace = ptxsim::ExecutionTracer::get_trace();

    INFO("step_warp picked PC=" << picked_pc);
    INFO("Lane 0 PC after: " << w->get_thread(0)->get_pc());
    INFO("Lane 0 is_blocked: " << ws.threads[0].is_blocked);
    INFO("Lane 0 is_active: " << ws.threads[0].is_active);
    INFO("Trace lane 0 PCs: ");
    for (const auto& e : trace.threads[0].entries) {
        INFO("  PC=" << e.pc << " instr=" << e.instruction_text);
    }

    // CORE ASSERTION: lane 0's st.shared at PC=0 must have been dispatched
    // (i.e. lane 0 must appear in the trace at PC=0).
    //
    // If the dispatch gate incorrectly classified PC=0 as a reconvergence
    // point (because the stale entry's reconv=0 matched), `check_and_block`
    // would block lane 0, the early-return path in `execute_warp_instruction`
    // would fire, and the tracer would have NO entry for lane 0 at PC=0.
    bool lane0_dispatched = false;
    for (const auto& e : trace.threads[0].entries) {
        if (e.pc == static_cast<uint32_t>(PC_ST_SHARED)) {
            lane0_dispatched = true;
            break;
        }
    }

    // This is a HYPOTHESIS test: if it passes, the stale entry did NOT block
    // the st.shared, meaning the bug is elsewhere. If it fails, the gate IS
    // the bug and we need to fix `check_and_block_at_reconvergence_point` to
    // ignore entries whose active_mask lanes have all passed the
    // reconvergence_pc.
    CHECK(lane0_dispatched);
    CHECK(picked_pc == PC_ST_SHARED);
    CHECK(!ws.threads[0].is_blocked);
}

// ============================================================================
// U-2: A NON-stale SIMTStackEntry with reconvergence_pc=X DOES block
// lane 0's instruction at PC=X (the EXPECTED, CORRECT behavior).
//
// This locks in the current "correct" behavior: the gate blocks lanes
// at a reconvergence_pc when other divergent lanes are still elsewhere.
// This is essential for SIMT semantics (don't run past the join point
// until everyone arrives). It is a complementary "definition" test.
// ============================================================================
TEST_CASE("U-2: non-stale SIMT entry with reconv=X blocks lane at PC=X",
          "[barrier][dispatch][simt][definition]")
{
    ptxsim::testing::init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    constexpr int PC_ST_SHARED = 0;
    constexpr int PC_OTHER     = 5;  // divergent lane 1 is at PC=5
    constexpr int PC_RET       = 1;
    constexpr int NUM_STMTS    = 2;

    std::vector<StatementContext> v(NUM_STMTS);
    for (auto& s : v) s = make_nop();
    v[PC_ST_SHARED] = ptxsim::testing::make_st_shared_addr(
        "sdata", "r_tid", "r_val", Qualifier::Q_B32);
    v[PC_RET] = make_ret();

    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1}, b{32, 1, 1}, bi{0, 0, 0};
    std::map<std::string, int> l2pc = {};
    std::map<std::string, std::unique_ptr<Symtable>> n2s;
    blk->init(g, b, bi, v, &n2s, l2pc);
    blk->sharedMemBytes = 1024;

    SMContext sm(4, 128, 4096, 0);
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    WarpContext* w = sm.get_warp(0);
    REQUIRE(w != nullptr);

    auto& ws = w->get_warp_state();
    ws.threads[0].pc = PC_ST_SHARED;
    ws.threads[0].next_pc = PC_ST_SHARED;
    ws.threads[0].is_active = true;
    ws.threads[0].is_blocked = false;
    ws.threads[0].is_exited = false;
    ws.threads[0].status = ptxsim::ThreadStatus::Active;
    for (int i = 1; i < 32; i++) {
        ws.threads[i].pc = PC_OTHER;  // divergent path
        ws.threads[i].next_pc = PC_OTHER;
        ws.threads[i].is_active = true;
        ws.threads[i].is_blocked = false;
        ws.threads[i].is_exited = false;
        ws.threads[i].status = ptxsim::ThreadStatus::Active;
    }
    w->update_active_mask();

    // *** Inject a NON-stale entry: lane 1 (active_mask=0xFFFFFFFE) is still
    // on a divergent path at PC=PC_OTHER != reconv_pc=PC_ST_SHARED ***
    ptxsim::SIMTStackEntry entry;
    entry.branch_pc = 999;
    entry.reconvergence_pc = PC_ST_SHARED;
    entry.active_mask = 0xFFFFFFFEu;  // lanes 1-31: divergent, NOT at reconv
    entry.return_mask = 0xFFFFFFFEu;
    entry.return_pc = PC_ST_SHARED;
    w->get_simt_stack().push(entry);

    ptxsim::ExecutionTracer::enable();
    ptxsim::ExecutionTracer::reset();

    int picked_pc = step_warp(w, v);

    ptxsim::ExecutionTracer::disable();
    const auto& trace = ptxsim::ExecutionTracer::get_trace();

    INFO("step_warp picked PC=" << picked_pc);
    INFO("Lane 0 is_blocked: " << ws.threads[0].is_blocked);

    // The dispatcher should have recognized the reconvergence point
    // and blocked lane 0 (because lane 1 is still on the divergent path).
    // lane 0's st.shared at PC=0 should NOT be dispatched.
    bool lane0_dispatched = false;
    for (const auto& e : trace.threads[0].entries) {
        if (e.pc == static_cast<uint32_t>(PC_ST_SHARED)) {
            lane0_dispatched = true;
            break;
        }
    }

    CHECK(!lane0_dispatched);     // gated: st.shared blocked
    CHECK(ws.threads[0].is_blocked);  // lane 0 blocked
}
