/**
 * @file test_broadcast_after_barrier.cpp
 * @brief Instruction-sequence integration test for BUG-CUTE-RMSNORM-BROADCAST-SKIP.
 *
 * Reproduces the cute_rmsnorm broadcast barrier pattern at full PTX level:
 * a divergent warp (one lane on a write path, the rest on a skip path)
 * reconverges at bar.warp.sync. After the barrier, all 32 lanes must
 * execute the broadcast `ld.shared` instruction at reconvergence_pc.
 *
 * Pattern (matches cute_rmsnorm.cu broadcast loop):
 *   PC=0: setp.eq.s32 p1, r_tid, 0
 *   PC=1: @p1 bra L_TID0
 *   PC=2: bra L_CONV
 *   PC=3: L_TID0: mov r1, 1            ; lane 0's fast path
 *   PC=4: L_CONV: bar.warp.sync 0xFFFFFFFF, 5
 *   PC=5: ld.shared.b32 r2, [sdata]    ; broadcast read (PROTECTED PC)
 *   PC=6: ret
 *
 * Symptom (per docs/developer-guide/KNOWN_ISSUES.md §"cute_rmsnorm —
 * broadcast-after-barrier skipped"):
 *   - BarWarpSyncHandler correctly releases all 32 lanes to PC=reconv_pc=5
 *   - But the broadcast `ld.shared.f32 %f8, [sdata]` at PC=5 is NEVER
 *     executed by the scheduler (LdHandler 0 hits in cute_rmsnorm trace)
 *   - Downstream `st.global.f32` then writes 0 to output
 *   - Result: output[0] = 0 (expected ≈ input[0] / rms)
 *
 * Test strategy: use ExecutionTracer to record every (lane, PC) executed
 * by execute_warp_instruction. After step_warp loop completes, verify
 * that EVERY lane has at least one trace entry at PC=5 (the broadcast
 * instruction). If the bug fires, the trace for lanes 1-31 lacks PC=5
 * (only PC=4 appears; the scheduler jumps straight from barrier to ret
 * or post-broadcast PC).
 *
 * This complements unit_broadcast_after_barrier (which proves the handler
 * itself releases to reconv_pc) by proving the SCHEDULER dispatches
 * reconv_pc after release.
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
#include <iostream>

using namespace ptxir::factory;
using ptxsim::testing::step_warp;
using ptxsim::testing::setup_pred;
using ptxsim::testing::make_nop;
using ptxsim::testing::make_bra;
using ptxsim::testing::make_bra_pred;
using ptxsim::testing::make_bar_warp_sync;
using ptxsim::testing::make_ret;
using ptxsim::testing::make_mov_imm;
using ptxsim::testing::make_setp_lt;
using ptxsim::testing::make_shared_decl;
using ptxsim::testing::make_st_shared_addr;
using ptxsim::testing::make_ld_shared_addr;
using ptxsim::testing::read_reg_u32;
using ptxsim::ExecutionTracer;
using ptxsim::ThreadStatus;

// ============================================================================
// Test 1: cute_rmsnorm broadcast pattern — every lane must execute
// the broadcast ld.shared after bar.warp.sync
// ============================================================================
//
// Instruction layout (kept minimal so the test reads clearly):
//   PC=0:  setp.eq.b32 p1, r_tid, 0
//   PC=1:  @p1 bra L_TID0              ; divergence starts here
//   PC=2:  bra L_CONV                  ; path B: skip the write
//   PC=3:  L_TID0: mov.u32 r1, 0xABCD  ; path A: lane 0's "write"
//   PC=4:  L_CONV: bar.warp.sync 0xFFFFFFFF, 5
//   PC=5:  ld.shared.b32 r2, [sdata+0] ; BROADCAST READ (protected PC)
//   PC=6:  ret
//
// setup_pred(w, 0x00000001) makes only lane 0 take the @p1 branch.
// All 32 lanes must converge at PC=4 (barrier), then the barrier releases
// them to PC=5 (broadcast ld.shared). With the bug, the scheduler
// advances lanes 1-31 from PC=4 directly to PC=6 (or beyond), skipping
// PC=5 entirely.
// ============================================================================
TEST_CASE("I-1: cute_rmsnorm broadcast pattern — every lane executes "
          "ld.shared after bar.warp.sync",
          "[barrier][broadcast][divergence][regression][integrated]"
          "[BUG-CUTE-RMSNORM-BROADCAST-SKIP]")
{
    ptxsim::testing::init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    // PC layout (kept as named constants for readability)
    constexpr int PC_SETP    = 0;
    constexpr int PC_BRA     = 1;
    constexpr int PC_BRA_SKP = 2;
    constexpr int PC_TID0    = 3;  // label L_TID0
    constexpr int PC_CONV    = 4;  // label L_CONV (= barrier PC)
    constexpr int PC_LDSH    = 5;  // broadcast ld.shared (THE PROTECTED PC)
    constexpr int PC_RET     = 6;
    constexpr int NUM_STMTS  = 7;

    std::vector<StatementContext> v(NUM_STMTS);
    for (auto& s : v) s = make_nop();

    // PC=0: setp.eq.b32 p1, r_tid, 0
    v[PC_SETP] = makeGenericInstr(
        S_SETP,
        {Qualifier::Q_B32, Qualifier::Q_EQ},
        {OperandContext{RegOperand{"p1", -1}},
         OperandContext{RegOperand{"r_tid", -1}},
         OperandContext{ImmOperand{"0"}}},
        "setp.eq.b32 %p1, %r_tid, 0;");

    // PC=1: @p1 bra L_TID0 (predicated branch — pushes SIMT stack entry)
    v[PC_BRA] = make_bra_pred("L_TID0", "p1", false, PC_CONV);

    // PC=2: bra L_CONV (unconditional — for lanes that didn't take the @p1 bra)
    v[PC_BRA_SKP] = make_bra("L_CONV");

    // PC=3: L_TID0: mov.u32 r1, 0xABCD (lane 0's path — symbolic "write")
    v[PC_TID0] = make_mov_imm("r1", 0xABCDu);

    // PC=4: L_CONV: bar.warp.sync 0xFFFFFFFF, 5 (release to PC=5)
    v[PC_CONV] = make_bar_warp_sync(0xFFFFFFFFu, PC_LDSH);

    // PC=5: ld.shared.b32 r2, [sdata+r_tid]  (broadcast-style read)
    v[PC_LDSH] = ptxsim::testing::make_ld_shared_addr(
        "r2", "sdata", "r_tid", Qualifier::Q_B32);

    // PC=6: ret
    v[PC_RET] = make_ret();

    // Build the block + warp
    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1}, b{32, 1, 1}, bi{0, 0, 0};
    std::map<std::string, int> l2pc = {
        {"L_TID0", PC_TID0},
        {"L_CONV", PC_CONV},
    };
    std::map<std::string, std::unique_ptr<Symtable>> n2s;
    blk->init(g, b, bi, v, &n2s, l2pc);
    blk->sharedMemBytes = 1024;

    SMContext sm(4, 128, 4096, 0);
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    WarpContext* w = sm.get_warp(0);
    REQUIRE(w != nullptr);

    // Pre-write the broadcast value at sdata[0] so the ld.shared can succeed.
    // Lane 0 (which "writes" in the divergent path) is expected to read this
    // back; lanes 1-31 also read it because their effective address is
    // sdata+lane_id, but the test focuses on the SCHEDULING question, not
    // the per-lane read result, so the exact byte pattern is unimportant.
    if (w->get_thread(0)) {
        auto& smem = w->get_thread(0)->shared_mem_space;
        if (smem) {
            *reinterpret_cast<uint32_t*>(smem) = 0xABCDu;
        }
    }

    // Make lane 0 take the @p1 bra (path A), lanes 1-31 fall through (path B)
    setup_pred(w, 0x00000001u);

    // Enable the execution tracer so we can verify each lane actually
    // executed PC=5 (the broadcast ld.shared) at some point.
    ptxsim::ExecutionTracer::enable();
    ptxsim::ExecutionTracer::reset();

    // Drive the warp until it finishes (or step budget is exhausted)
    constexpr int MAX_STEPS = 50;
    for (int i = 0; i < MAX_STEPS; i++) {
        if (w->is_finished()) break;
        step_warp(w, v);
    }

    ptxsim::ExecutionTracer::disable();
    const auto& trace = ptxsim::ExecutionTracer::get_trace();

    // CORE ASSERTION: every lane must have executed PC=PC_LDSH (=5) at least
    // once. If the broadcast-after-barrier skip bug fired, lanes 1-31 will
    // be missing the PC=5 trace entry — they jumped from PC=4 (barrier)
    // straight to PC=6 (ret).
    INFO("Traced " << MAX_STEPS << " steps; verifying broadcast instruction "
         "executed by all 32 lanes");
    for (int lane = 0; lane < 32; lane++) {
        const auto& entries = trace.threads[lane].entries;
        bool saw_broadcast = false;
        std::string pclist;
        for (const auto& e : entries) {
            if (!pclist.empty()) pclist += ",";
            pclist += std::to_string(e.pc);
            if (e.pc == static_cast<uint32_t>(PC_LDSH)) {
                saw_broadcast = true;
            }
        }
        INFO("Lane " << lane << " executed PCs: [" << pclist << "]");
        CHECK(saw_broadcast);
    }
}

// ============================================================================
// Test 2: cute_rmsnorm reduction loop + broadcast barrier — exact reproduction
// ============================================================================
//
// This test reproduces the cute_rmsnorm.cu broadcast loop structure more
// faithfully. The PTX has 3 distinct phases:
//   1. Reduction loop: multiple iterations of st.shared + bar.warp.sync
//      (sdata[s] = sdata[s] + sdata[s + half_size]) with half_size halving
//      each iteration. In cute_rmsnorm PTX, this is the s = blockSize/2
//      loop that runs 8 times for kBlockSize=256.
//   2. Single-lane write: only lane 0 (tid==0) writes sdata[0] = rsqrt(...).
//   3. Broadcast: bar.warp.sync followed by all 32 lanes reading sdata[0].
//
// The bug fires in PHASE 3: after the bar.warp.sync in PHASE 3, lanes 1-31
// skip the broadcast ld.shared and the downstream st.global writes 0.
//
// We collapse the 8 reduction iterations into 2-3 iterations to keep the
// test small while still triggering the divergent SIMT-stack pattern that
// cute_rmsnorm exhibits.
// ============================================================================
TEST_CASE("I-2: cute_rmsnorm reduction loop + broadcast barrier — full pattern",
          "[barrier][broadcast][divergence][regression][integrated]"
          "[BUG-CUTE-RMSNORM-BROADCAST-SKIP]")
{
    ptxsim::testing::init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    // PC layout (each reduction iteration is 4 statements: setp_lt, bra,
    // st.shared, bar.warp.sync, then back-edge bra; lane-0 write is 3;
    // broadcast is 3; ret is 1)
    //
    //   PC=0:  setp.lt.b32 p1, r_tid, 1     ; first iteration: s=1, only lane 0
    //   PC=1:  @p1 bra L_WR1
    //   PC=2:  bra L_BAR1
    //   PC=3:  L_WR1: st.shared.b32 [sdata+r_tid], r_val
    //   PC=4:  L_BAR1: bar.warp.sync 0xFFFFFFFF, 5
    //   PC=5:  setp.lt.b32 p2, r_tid, 1     ; (no second iteration in this test)
    //   PC=6:  @p2 bra L_WR2
    //   PC=7:  bra L_BAR2
    //   PC=8:  L_WR2: st.shared.b32 [sdata+r_tid], r_val
    //   PC=9:  L_BAR2: bar.warp.sync 0xFFFFFFFF, 10
    //   PC=10: setp.eq.b32 p3, r_tid, 0     ; divergence: only lane 0 writes
    //   PC=11: @p3 bra L_TID0_W
    //   PC=12: bra L_BCONV
    //   PC=13: L_TID0_W: st.shared.b32 [sdata+0], r_rsqrt
    //   PC=14: L_BCONV: bar.warp.sync 0xFFFFFFFF, 15
    //   PC=15: ld.shared.b32 r2, [sdata+0]  ; ← BROADCAST READ
    //   PC=16: ret
    //
    // The key PC is 15 (the broadcast ld.shared). The bug causes lanes
    // 1-31 to skip PC=15 entirely after the PC=14 bar.warp.sync.
    constexpr int NUM_STMTS = 17;
    constexpr int PC_BCONV  = 14;
    constexpr int PC_LDSH   = 15;

    std::vector<StatementContext> v(NUM_STMTS);
    for (auto& s : v) s = make_nop();

    // ----- Iteration 1 (s=1, only lane 0) -----
    v[0] = make_setp_lt("p1", "r_tid", "1");
    v[1] = make_bra_pred("L_WR1", "p1", false, /*reconv*/ 2);
    v[2] = make_bra("L_BAR1");
    v[3] = ptxsim::testing::make_st_shared_addr(
        "sdata", "r_tid", "r_val", Qualifier::Q_B32);
    v[4] = make_bar_warp_sync(0xFFFFFFFFu, /*reconv*/ 5);

    // ----- Iteration 2 (also s=1, only lane 0 — same predicate for simplicity) -----
    v[5] = make_setp_lt("p2", "r_tid", "1");
    v[6] = make_bra_pred("L_WR2", "p2", false, /*reconv*/ 7);
    v[7] = make_bra("L_BAR2");
    v[8] = ptxsim::testing::make_st_shared_addr(
        "sdata", "r_tid", "r_val", Qualifier::Q_B32);
    v[9] = make_bar_warp_sync(0xFFFFFFFFu, /*reconv*/ 10);

    // ----- Lane-0 write (rsqrt result) -----
    v[10] = makeGenericInstr(
        S_SETP,
        {Qualifier::Q_B32, Qualifier::Q_EQ},
        {OperandContext{RegOperand{"p3", -1}},
         OperandContext{RegOperand{"r_tid", -1}},
         OperandContext{ImmOperand{"0"}}},
        "setp.eq.b32 %p3, %r_tid, 0;");
    v[11] = make_bra_pred("L_TID0_W", "p3", false, /*reconv*/ 12);
    v[12] = make_bra("L_BCONV");
    v[13] = ptxsim::testing::make_st_shared_addr(
        "sdata", "r_tid", "r_rsqrt", Qualifier::Q_B32);

    // ----- Broadcast barrier + read -----
    v[PC_BCONV] = make_bar_warp_sync(0xFFFFFFFFu, /*reconv*/ PC_LDSH);
    v[PC_LDSH]  = ptxsim::testing::make_ld_shared_addr(
        "r2", "sdata", "r_tid", Qualifier::Q_B32);
    v[PC_LDSH + 1] = make_ret();

    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1}, b{32, 1, 1}, bi{0, 0, 0};
    std::map<std::string, int> l2pc = {
        {"L_WR1", 3}, {"L_BAR1", 4},
        {"L_WR2", 8}, {"L_BAR2", 9},
        {"L_TID0_W", 13}, {"L_BCONV", PC_BCONV},
    };
    std::map<std::string, std::unique_ptr<Symtable>> n2s;
    blk->init(g, b, bi, v, &n2s, l2pc);
    blk->sharedMemBytes = 1024;

    SMContext sm(4, 128, 4096, 0);
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    WarpContext* w = sm.get_warp(0);
    REQUIRE(w != nullptr);

    // All lanes: setp says tid < 1, so only lane 0 is active in iter 1 & 2
    // (the reductions). Then in lane-0 write, p3 is true only for lane 0.
    // This is the cute_rmsnorm pattern.
    setup_pred(w, 0x00000001u);  // pred1/pred2/pred3 all start the same
    ptxsim::ExecutionTracer::enable();
    ptxsim::ExecutionTracer::reset();

    constexpr int MAX_STEPS = 80;
    for (int i = 0; i < MAX_STEPS; i++) {
        if (w->is_finished()) break;
        step_warp(w, v);
    }

    ptxsim::ExecutionTracer::disable();
    const auto& trace = ptxsim::ExecutionTracer::get_trace();

    // CORE ASSERTION: every lane must have executed PC=PC_LDSH (=15) at least
    // once. If the broadcast-after-barrier skip bug fired, lanes 1-31 will
    // be missing the PC=15 trace entry — they jumped from PC=14 (barrier)
    // straight to PC=16 (ret).
    for (int lane = 0; lane < 32; lane++) {
        const auto& entries = trace.threads[lane].entries;
        bool saw_broadcast = false;
        std::string pclist;
        for (const auto& e : entries) {
            if (!pclist.empty()) pclist += ",";
            pclist += std::to_string(e.pc);
            if (e.pc == static_cast<uint32_t>(PC_LDSH)) {
                saw_broadcast = true;
            }
        }
        INFO("Lane " << lane << " executed PCs: [" << pclist << "]");
        CHECK(saw_broadcast);
    }
}
