// tests/unit/barrier/test_broadcast_after_barrier.cpp
//
// Unit test for BUG-CUTE-RMSNORM-BROADCAST-SKIP.
//
// Bug: When a divergent warp (one lane on a write path, the rest on a skip
// path) reconverges at a `bar.warp.sync` (broadcast barrier), the barrier
// correctly releases all 32 lanes to `reconvergence_pc`. However, the
// `ld.shared.f32` broadcast instruction at `reconvergence_pc` is NEVER
// executed by the scheduler — the warp effectively skips from the barrier
// PC to the PC AFTER the broadcast load.
//
// Symptom (cute_rmsnorm trace):
//   st.shared.f32 [sdata], %f29   at PC=91   ✅ executed
//   bar.warp.sync                 at PC=108  ✅ releases 32 lanes to PC=109
//   ld.shared.f32 %f8, [sdata]   at PC=109  ❌ never called (LdHandler 0 hits)
//   st.global.f32                 at PC=134  ✅ executed (src_val=0 → output=0)
//
// Reproduction pattern (matches cute_rmsnorm.cu broadcast loop):
//   PC=0: setp.eq.s32 p1, tid, 0
//   PC=1: @p1 bra L_TID0
//   PC=2: bra L_CONV
//   PC=3: L_TID0: st.shared.f32 [sdata], r1
//   PC=4: L_CONV: bar.warp.sync 0xFFFFFFFF, 5
//   PC=5: ld.shared.f32 r2, [sdata]   ← broadcast read; BUG skips this
//   PC=6: ret
//
// The fix (planned, not yet implemented per KNOWN_ISSUES.md §"cute_rmsnorm"):
// Rewrite the `force_reconvergence_at_barrier` / `advance_thread_pc`
// interaction in src/ptxsim/instructions/barrier.cpp:150-220 so that
// released lanes reliably land on `reconvergence_pc` (not `reconvergence_pc
// + 1`) and the scheduler dispatches the broadcast load before advancing
// past it.

#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/testing/memory_test_utils.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_factory.h"
#include "ptxsim/cta_context.h"
#include <memory>
#include <vector>
#include <map>

using namespace ptxsim;
using ptxsim::ThreadStatus;

namespace {

// Add a thread at `lane` to the warp. The thread is initialized to RUN state
// and points at a real CTAContext's statements vector. We only use it to
// satisfy WarpContext::add_thread's contract — the test does not execute
// real instructions through it.
void add_thread(WarpContext& warp, int lane) {
    auto thread = std::make_unique<ThreadContext>();
    Dim3 blockIdx = {0, 0, 0};
    Dim3 threadIdx = {(uint32_t)lane, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<StatementContext> stmts;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    thread->init(blockIdx, threadIdx, gridDim, blockDim, stmts, &name2Sym,
                 label2pc, nullptr, nullptr);
    thread->set_state(RUN);
    warp.add_thread(std::move(thread), lane);
}

// Set up a divergent warp matching the cute_rmsnorm broadcast pattern:
//   lane 0:   on path A (PC=A)  — would write sdata[0] in real code
//   lanes 1-31: on path B (PC=B)  — skip the write
//   all 32 lanes converge at PC=BARRIER_PC (the bar.warp.sync)
//
// Returns nothing; mutates warp_state in place.
void setup_divergent_warp(WarpContext& warp, int path_a_pc, int path_b_pc) {
    auto& ws = warp.get_warp_state();
    for (int i = 0; i < 32; i++) {
        ws.threads[i].pc = (i == 0) ? path_a_pc : path_b_pc;
        ws.threads[i].is_active = true;
        ws.threads[i].is_blocked = false;
        ws.threads[i].is_exited = false;
        ws.threads[i].status = ThreadStatus::Active;
    }
    warp.set_active_mask(0xFFFFFFFFu);
    warp.set_exec_mask(0xFFFFFFFFu);
}

}  // namespace

// =============================================================================
// U-1: divergent warp releases to reconvergence_pc — no lane may land PAST it
// =============================================================================
//
// This is the smallest possible reproduction of the bug. We arrange a divergent
// warp (one lane on a fast path, the other 31 on a skip path) and call
// BarWarpSyncHandler::processOperation as if executing the barrier from
// lane 0's perspective. The handler's force_reconvergence branch must:
//   1. Initialize the wbar with participation_mask covering ALL 32 lanes
//   2. Mark lane 0 as arrived
//   3. NOT complete the barrier (only 1/32 lanes have arrived)
//   4. Block lane 0
//
// When the remaining 31 lanes arrive (simulated), the handler should
// complete the barrier and release ALL 32 lanes to `reconvergence_pc`
// (which is the PC of the broadcast `ld.shared.f32`). No lane may be set
// to `reconvergence_pc + 1` (the post-broadcast instruction), because the
// broadcast load is the very instruction the barrier exists to protect.
// =============================================================================
TEST_CASE("U-1: divergent warp broadcast barrier — all 32 lanes released to "
          "reconvergence_pc, not reconvergence_pc+1",
          "[barrier][broadcast][regression][BUG-CUTE-RMSNORM-BROADCAST-SKIP]")
{
    ptxsim::testing::init_instruction_factory_once();

    constexpr int PATH_A_PC    = 5;   // lane 0's fast path
    constexpr int PATH_B_PC    = 10;  // lanes 1-31's skip path
    constexpr int BARRIER_PC   = 12;  // bar.warp.sync location
    constexpr int RECONV_PC    = 13;  // broadcast ld.shared (the protected PC)

    // Build a minimal statement list that includes the barrier at the right PC
    std::vector<StatementContext> stmts(BARRIER_PC + 1);
    for (auto& s : stmts) s = ptxir::factory::makeVoidInstr(S_PRAGMA, "nop;");
    stmts[BARRIER_PC] = ptxir::factory::makeBarWarpSyncInstr(
        0xFFFFFFFFu, RECONV_PC,
        "bar.warp.sync.b32 0xFFFFFFFF, " + std::to_string(RECONV_PC) + ";");

    // Build a real WarpContext attached to a CTA so the barrier handler's
    // sm_context_ lookup doesn't NPE (it queries the SM context's bsync_manager
    // and may also call advance_thread_pc which expects real warp state).
    WarpContext warp;
    auto cta = std::make_unique<CTAContext>();
    cta->ensure_barrier_module();
    CTAContext* cta_ptr = cta.get();
    warp.set_cta_context(cta_ptr);
    for (int i = 0; i < 32; i++) add_thread(warp, i);
    setup_divergent_warp(warp, PATH_A_PC, PATH_B_PC);

    // Drive the barrier by directly executing the barrier handler on each
    // lane, since step_warp would require a SMContext to dispatch.
    //
    // The handler is per-thread; we invoke it for lane 0 first (path A
    // arrives), then for the remaining 31 lanes (path B arrives). This
    // mirrors what the scheduler would do across two step_warp calls.
    auto* handler = InstructionFactory::get_handler(S_BAR_WARP_SYNC);
    REQUIRE(handler != nullptr);

    // Lane 0 arrives: divergent, so force_reconvergence path is taken
    warp.get_warp_state().threads[0].pc = BARRIER_PC;
    {
        ThreadContext* t = warp.get_thread(0);
        t->sync_from_warp_state();
        handler->ExecPipe(t, stmts[BARRIER_PC]);
        t->sync_to_warp_state();
    }

    // After lane 0's arrival: wbar should be initialized for the FULL warp
    // (static_mask = 0xFFFFFFFF), not just the partial dynamic mask.
    auto* wbar = cta_ptr->get_barrier_module().get_warp_barrier(0);
    REQUIRE(wbar != nullptr);
    REQUIRE(wbar->is_initialized());
    CHECK(wbar->get_participation_mask() == 0xFFFFFFFFu);
    CHECK(wbar->get_expected_count() == 32);
    CHECK(wbar->get_arrived_count() == 1);
    CHECK(!wbar->is_complete());

    // Lanes 1-31 arrive: each calls the barrier handler from PC=BARRIER_PC
    for (int lane = 1; lane < 32; lane++) {
        warp.get_warp_state().threads[lane].pc = BARRIER_PC;
        ThreadContext* t = warp.get_thread(lane);
        t->sync_from_warp_state();
        handler->ExecPipe(t, stmts[BARRIER_PC]);
        t->sync_to_warp_state();
    }

    // =========================================================================
    // CORE ASSERTION: ALL 32 lanes must be released to RECONV_PC (=13),
    // not RECONV_PC+1 (=14). The bug releases them to RECONV_PC+1 (or to
    // some PC > RECONV_PC), skipping the broadcast ld.shared.f32 entirely.
    // =========================================================================
    auto& ws = warp.get_warp_state();
    for (int i = 0; i < 32; i++) {
        INFO("Lane " << i << " pc=" << ws.threads[i].pc
             << " next_pc=" << ws.threads[i].next_pc);
        CHECK(ws.threads[i].pc == RECONV_PC);
        CHECK(ws.threads[i].next_pc == RECONV_PC);
        CHECK(!ws.threads[i].is_blocked);
    }
    CHECK(!warp.get_cta_context()->get_barrier_module().get_warp_barrier(0)->is_initialized());
    // All 32 lanes should be schedulable at RECONV_PC for the broadcast load
    CHECK(warp.is_warp_ready_to_fetch());
}

// =============================================================================
// U-2: divergent warp — first-arriving lane on a partial path is also OK
// =============================================================================
//
// Variant: lane 0 first arrives via path A, then the other 31 arrive
// "consecutively" via path B (force_reconvergence re-init'd by the first
// call). The barrier must still complete and release all 32 lanes to
// RECONV_PC. This exercises the `else { participation_mask = static_mask; }`
// branch in barrier.cpp:172-176 (the BUG-RECONVERGENCE-SIMPLEGEMM fix path).
// =============================================================================
TEST_CASE("U-2: divergent warp — lane 0 arrives alone, then 31 arrive in bulk",
          "[barrier][broadcast][divergence][regression]"
          "[BUG-CUTE-RMSNORM-BROADCAST-SKIP]")
{
    ptxsim::testing::init_instruction_factory_once();

    constexpr int PATH_A_PC  = 5;
    constexpr int PATH_B_PC  = 10;
    constexpr int BARRIER_PC = 12;
    constexpr int RECONV_PC  = 13;

    std::vector<StatementContext> stmts(BARRIER_PC + 1);
    for (auto& s : stmts) s = ptxir::factory::makeVoidInstr(S_PRAGMA, "nop;");
    stmts[BARRIER_PC] = ptxir::factory::makeBarWarpSyncInstr(
        0xFFFFFFFFu, RECONV_PC,
        "bar.warp.sync.b32 0xFFFFFFFF, " + std::to_string(RECONV_PC) + ";");

    WarpContext warp;
    auto cta = std::make_unique<CTAContext>();
    cta->ensure_barrier_module();
    CTAContext* cta_ptr = cta.get();
    warp.set_cta_context(cta_ptr);
    for (int i = 0; i < 32; i++) add_thread(warp, i);
    setup_divergent_warp(warp, PATH_A_PC, PATH_B_PC);

    auto* handler = InstructionFactory::get_handler(S_BAR_WARP_SYNC);
    REQUIRE(handler != nullptr);

    // Phase 1: lane 0 arrives, handler takes the divergent path and
    // initializes the wbar with static_mask=0xFFFFFFFF.
    warp.get_warp_state().threads[0].pc = BARRIER_PC;
    {
        ThreadContext* t = warp.get_thread(0);
        t->sync_from_warp_state();
        handler->ExecPipe(t, stmts[BARRIER_PC]);
        t->sync_to_warp_state();
    }
    {
        auto* wbar = cta_ptr->get_barrier_module().get_warp_barrier(0);
        REQUIRE(wbar != nullptr);
        REQUIRE(wbar->is_initialized());
        CHECK(wbar->get_participation_mask() == 0xFFFFFFFFu);
        CHECK(wbar->get_expected_count() == 32);
        CHECK(wbar->get_arrived_count() == 1);
    }

    // Phase 2: all 31 other lanes arrive at once. The 32nd arrival completes
    // the barrier and must release ALL 32 lanes to RECONV_PC.
    for (int lane = 1; lane < 32; lane++) {
        warp.get_warp_state().threads[lane].pc = BARRIER_PC;
        ThreadContext* t = warp.get_thread(lane);
        t->sync_from_warp_state();
        handler->ExecPipe(t, stmts[BARRIER_PC]);
        t->sync_to_warp_state();
    }

    auto& ws = warp.get_warp_state();
    for (int i = 0; i < 32; i++) {
        INFO("Lane " << i << " pc=" << ws.threads[i].pc);
        CHECK(ws.threads[i].pc == RECONV_PC);
        CHECK(ws.threads[i].next_pc == RECONV_PC);
    }
    // Critical: the broadcast instruction at RECONV_PC must be schedulable
    // for all 32 lanes. If the bug fired, the lanes would be at RECONV_PC+1
    // or beyond, and the broadcast load would be skipped.
    CHECK(warp.get_active_count() == 32);
    CHECK(warp.is_warp_ready_to_fetch());
}
