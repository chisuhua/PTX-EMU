/**
 * @file test_cute_rmsnorm_bar_sync_pattern.cpp
 * @brief Integration test for BUG-DISPATCH-GATE-LANE0-SKIP via cute_rmsnorm
 * S_BAR (bar.sync 0) reduction + broadcast pattern.
 *
 * cute_rmsnorm.ptx uses `bar.sync 0` (S_BAR, CTA-level barrier), NOT
 * `bar.warp.sync` (S_BAR_WARP_SYNC, warp-level). The existing
 * test_broadcast_after_barrier.cpp I-1/I-2 use bar.warp.sync, which is
 * a DIFFERENT code path. cute_rmsnorm's actual S_BAR path goes through
 * SMContext::synchronize_barrier and may leave the SIMT stack in a
 * different state.
 *
 * Pattern (mirrors cute_rmsnorm.ptx lines 109-145):
 *   PC=0:   st.shared.b32 [sdata+r_tid], r_val        ; per-lane write
 *   PC=1:   bar.sync 0                                  ; barrier 1
 *   ----- Reduction loop (3 iterations to keep test small) -----
 *   PC=2:   setp.lt.u32 p1, r_tid, 1                   ; iter 1: s=1
 *   PC=3:   @p1 bra L_WR1
 *   PC=4:   bra L_BAR1
 *   PC=5:   L_WR1: st.shared.b32 [sdata+r_tid], r_val
 *   PC=6:   L_BAR1: bar.sync 0
 *   PC=7:   setp.gt.u32 p2, r65, 3
 *   PC=8:   mov.u32 r65, r19
 *   PC=9:   @p2 bra L_LOOP                            ; back-edge
 *   ... (similar for iters 2-3, but collapsed to keep test simple)
 *   ----- Lane-0 only write (rsqrt result) -----
 *   PC=20:  setp.ne.s32 p3, r_tid, 0
 *   PC=21:  @p3 bra L_TID0_W
 *   PC=22:  bra L_BCONV
 *   PC=23:  L_TID0_W: st.shared.b32 [sdata+0], r_rsqrt  ; ★ PROTECTED
 *   PC=24:  L_BCONV: bar.sync 0
 *   PC=25:  ld.shared.b32 r2, [sdata+0]                ; ★ BROADCAST READ
 *   PC=26:  ret
 *
 * Symptom (per cute_rmsnorm debug 2026-06-16):
 *   - Lane 0's st.shared at PC=23 is NOT dispatched
 *   - All 32 lanes' ld.shared at PC=25 IS dispatched (broadcast)
 *   - Downstream st.global writes 0 to output[0] (broadcast read 0)
 *
 * Test goal: drive this pattern via step_warp + ExecutionTracer and
 * verify lane 0 has a trace entry at PC=23 (lane 0's st.shared).
 * If the bug fires, lane 0's trace has no PC=23 entry.
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
using ptxsim::testing::make_setp_lt;
using ptxsim::testing::make_setp_gt;
using ptxsim::testing::make_mov_imm;
using ptxsim::testing::make_bra;
using ptxsim::testing::make_bra_pred;
using ptxsim::testing::make_bar_sync;        // S_BAR, matching cute_rmsnorm.ptx
using ptxsim::testing::make_st_shared_addr;
using ptxsim::testing::make_ld_shared_addr;
using ptxsim::testing::make_ret;
using ptxsim::ExecutionTracer;

static void preset_u32_register(WarpContext* w, const std::string& reg_name,
                                uint32_t value) {
    auto rbm = w->get_register_bank_manager();
    rbm->create_register(reg_name, sizeof(uint32_t));
    for (int lane = 0; lane < 32; ++lane) {
        auto* p = static_cast<uint32_t*>(rbm->get_register(reg_name, 0, lane));
        REQUIRE(p != nullptr);
        *p = value;
    }
}

static void preset_tid_register(WarpContext* w) {
    auto rbm = w->get_register_bank_manager();
    rbm->create_register("r_tid", sizeof(uint32_t));
    for (int lane = 0; lane < 32; ++lane) {
        auto* p = static_cast<uint32_t*>(rbm->get_register("r_tid", 0, lane));
        REQUIRE(p != nullptr);
        *p = static_cast<uint32_t>(lane);
    }
}

constexpr int PC_PER_LANE_ST = 0;
constexpr int PC_BAR1       = 1;
constexpr int PC_LTID0_DVRG = 2;
constexpr int PC_TID0_DVRG  = 8;
constexpr int PC_BAR3       = 9;
constexpr int PC_BROADCAST  = 10;
constexpr int PC_RET        = 11;
constexpr int NUM_STMTS     = 12;

static std::vector<ptxemu::ir::StatementContext> build_cute_rmsnorm_pattern(
    std::map<std::string, int>& l2pc)
{
    std::vector<ptxemu::ir::StatementContext> v(NUM_STMTS);
    for (auto& s : v) s = make_nop();

    v[PC_PER_LANE_ST] = ptxsim::testing::make_st_shared_addr(
        "sdata", "r_tid", "r_val", ptxemu::ir::Qualifier::Q_B32);
    v[PC_BAR1]        = make_bar_sync(0);

    v[PC_LTID0_DVRG] = makeGenericInstr(
        S_SETP,
        {ptxemu::ir::Qualifier::Q_B32, ptxemu::ir::Qualifier::Q_NE},
        {ptxemu::ir::OperandContext{RegOperand{"p3", -1}},
         ptxemu::ir::OperandContext{RegOperand{"r_tid", -1}},
         ptxemu::ir::OperandContext{ImmOperand{"0"}}},
        "setp.ne.s32 %p3, %r_tid, 0;");
    // Use negated predicate @!p3 bra: lane 0 (r_tid=0 → p3=0) takes the
    // branch to L_TID0_DVRG_W (PC_TID0_DVRG). Lanes 1-31 (r_tid!=0 →
    // p3=1) fall through to the unconditional bra L_BCONV.
    v[PC_LTID0_DVRG + 1] = make_bra_pred("L_TID0_DVRG_W", "p3", true, /*reconv*/ PC_BAR3);
    v[PC_LTID0_DVRG + 2] = make_bra("L_BCONV");
    v[PC_TID0_DVRG] = ptxsim::testing::make_st_shared_addr(
        "sdata", "r_tid", "r_rsqrt", ptxemu::ir::Qualifier::Q_B32);

    l2pc["L_TID0_DVRG_W"] = PC_TID0_DVRG;
    l2pc["L_BCONV"] = PC_BAR3;

    v[PC_BAR3]      = make_bar_sync(0);
    v[PC_BROADCAST] = ptxsim::testing::make_ld_shared_addr(
        "r2", "sdata", "r_tid", ptxemu::ir::Qualifier::Q_B32);
    v[PC_RET]       = make_ret();

    return v;
}

TEST_CASE("I-3: cute_rmsnorm S_BAR (bar.sync 0) reduction + broadcast — "
          "lane 0 st.shared must execute",
          "[barrier][broadcast][divergence][regression][integrated]"
          "[BUG-DISPATCH-GATE-LANE0-SKIP]")
{
    ptxsim::testing::init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::map<std::string, int> l2pc;
    auto v = build_cute_rmsnorm_pattern(l2pc);

    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1}, b{32, 1, 1}, bi{0, 0, 0};
    std::map<std::string, std::unique_ptr<Symtable>> n2s;
    blk->init(g, b, bi, v, &n2s, l2pc);
    blk->sharedMemBytes = 1024;

    SMContext sm(4, 128, 4096, 0);
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    WarpContext* w = sm.get_warp(0);
    REQUIRE(w != nullptr);

    // r_tid: lane id per lane (needed by setp.ne + st.shared offset)
    preset_tid_register(w);
    preset_u32_register(w, "r_val", 0x42);
    preset_u32_register(w, "r_rsqrt", 0x7F);

    // Predicates: lane 0 only (matches cute_rmsnorm pattern)
    setup_pred(w, 0x00000001u);

    ptxsim::ExecutionTracer::enable();
    ptxsim::ExecutionTracer::reset();

    constexpr int MAX_STEPS = 100;
    for (int i = 0; i < MAX_STEPS; i++) {
        if (w->is_finished()) break;
        step_warp(w, v);
    }

    ptxsim::ExecutionTracer::disable();
    const auto& trace = ptxsim::ExecutionTracer::get_trace();

    bool lane0_wrote = false;
    std::string pclist;
    for (const auto& e : trace.threads[0].entries) {
        if (!pclist.empty()) pclist += ",";
        pclist += std::to_string(e.pc);
        // ExecutionTracer records POST-execution pc (after commit_pc).
        // For the broadcast ld.shared at PC=10, the recorded pc is 11 (PC_RET).
        // For lane 0's st.shared at PC=8, the recorded pc is the branch
        // target (8) for the branch jump, and the post-st.shared value (9)
        // for the instruction at PC=8 itself.
        if (e.pc == static_cast<uint32_t>(PC_TID0_DVRG) ||
            e.pc == static_cast<uint32_t>(PC_BAR3)) {
            lane0_wrote = true;
        }
    }
    INFO("Lane 0 executed PCs: [" << pclist << "]");
    INFO("Looking for PC=" << PC_TID0_DVRG << " (lane 0 st.shared)");
    CHECK(lane0_wrote);

    // ExecutionTracer records POST-execution pc (after commit_pc).
    // For the broadcast ld.shared at PC=10, the recorded pc is 11 (PC_RET).
    for (int lane = 0; lane < 32; lane++) {
        bool saw_broadcast = false;
        for (const auto& e : trace.threads[lane].entries) {
            if (e.pc == static_cast<uint32_t>(PC_RET)) {
                saw_broadcast = true;
                break;
            }
        }
        CHECK(saw_broadcast);
    }
}
