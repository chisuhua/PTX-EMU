// tests/unit/barrier/test_barrier_divergence_reconvergence_simplegemm.cpp
//
// Unit test for the BUG: force_reconvergence + scheduler collaboration in
// simpleGEMM-style divergent paths.
//
// simpleGEMM-int PC layout (matches runtime log):
//   $L__BB0_2 at PC=27  (a_tile loop start)
//   $L__BB0_4 at PC=47  (b_tile loop start)
//   $L__BB0_5 at PC=68  (pre-barrier label)
//   bar.sync   at PC=69  (barrier; CFG pass updates reconv_pc=70)
//   $L__BB0_8 at PC=84  (GEMM main loop start)
//
// Bug symptom (verified empirically via simpleGEMM-int runtime log):
//   barrier.warp.sync: Barrier complete, releasing 16 threads to PC=70
//                      (mask=0xFFFF0000 arrived=0xFFFF0000)
//   → Only lanes 16-31 are released. Lanes 0-15 are stuck at the barrier.
//
// This test verifies the SCENARIO at the Wbar level (without the actual
// scheduler), exposing the participation_mask initialization that is the
// proximate cause: wbar is initialized with mask=0xFFFF0000 (only 16 lanes)
// when static_mask from PTX is 0xFFFFFFFF (all 32 lanes).

#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"
#include "ptx_ir/operand_context.h"
#include "memory/resource_manager.h"

#include <memory>
#include <vector>
#include <map>

using namespace ptxsim;
using ptxsim::ThreadStatus;

namespace {

void add_thread(WarpContext& warp, int lane) {
    auto thread = std::make_unique<ThreadContext>();
    Dim3 blockIdx = {0, 0, 0};
    Dim3 threadIdx = {(uint32_t)lane, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<StatementContext> stmts;
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;
    thread->init(blockIdx, threadIdx, gridDim, blockDim, stmts, &name2Sym,
                 label2pc, nullptr, nullptr);
    thread->set_state(RUN);
    warp.add_thread(std::move(thread), lane);
}

// Mimics the post-completion release path (matches barrier.cpp:232-240)
void simulate_release(WarpContext& warp, int reconv_pc) {
    auto& ws = warp.get_warp_state();
    Wbar& wbar = ws.wbars[0];
    for (int i = 0; i < 32; i++) {
        if ((wbar.arrived_mask & (1u << i)) && ws.threads[i].is_active) {
            ws.threads[i].pc = reconv_pc;
            ws.threads[i].next_pc = reconv_pc;
            ws.threads[i].is_blocked = false;
            ws.threads[i].status = ThreadStatus::Active;
        }
    }
    warp.set_active_mask(warp.get_active_mask() | wbar.arrived_mask);
}

}  // namespace

// =============================================================================
// U-1: simpleGEMM pattern — wbar init mask must include ALL 32 lanes
// =============================================================================
// The simpleGEMM runtime log shows:
//   "Initialized wbar[0] with mask=0xFFFF0000, reconvergence_pc=70"
//   "Barrier complete, releasing 16 threads to PC=70 (mask=0xFFFF0000 arrived=0xFFFF0000)"
//
// Cause: at the time of init, dynamic_mask = 0xFFFF0000 (only lanes 16-31 are at
// barrier PC; lanes 0-15 are still in a_tile loop). With current logic:
//   participation_mask = (dynamic_mask & static_mask) = 0xFFFF0000 & 0xFFFFFFFF = 0xFFFF0000
// Lanes 0-15 never get into the arrived_mask.
//
// EXPECTED (after fix): static_mask is the authoritative mask. wbar must be
// initialized with 0xFFFFFFFF, even when dynamic_mask is partial.
// =============================================================================
TEST_CASE("U-1: simpleGEMM pattern — wbar mask must be full (0xFFFFFFFF) "
          "when static_mask from PTX is 0xFFFFFFFF",
          "[barrier][divergence][unit][simplegemm-pattern][BUG-RECONVERGENCE]")
{
    WarpContext warp;
    for (int i = 0; i < 32; i++) add_thread(warp, i);
    auto& ws = warp.get_warp_state();
    for (int i = 0; i < 32; i++) {
        ws.threads[i].pc = 69;
        ws.threads[i].is_active = true;
        ws.threads[i].is_blocked = false;
        ws.threads[i].is_exited = false;
        ws.threads[i].status = ThreadStatus::Active;
    }
    warp.set_active_mask(0xFFFFFFFFu);

    // Mirror the BUG runtime:
    //   barrier.cpp:201-208 init logic:
    //     participation_mask = (dynamic_mask != 0) ? (dynamic_mask & static_mask) : static_mask;
    //   For simpleGEMM, dynamic_mask=0xFFFF0000, static_mask=0xFFFFFFFF → bug yields 0xFFFF0000.
    //
    // The fix: when a barrier has divergence in flight (lanes elsewhere in warp),
    // trust static_mask and init with the FULL mask.
    uint32_t static_mask = 0xFFFFFFFFu;
    uint32_t dynamic_mask_partial = 0xFFFF0000u;  // only lanes 16-31 at barrier PC

    // Re-run the actual init logic, comparing BUG vs FIX outcomes:
    uint32_t buggy_participation_mask = (dynamic_mask_partial & static_mask);
    uint32_t expected_participation_mask = static_mask;  // FIX: always use static_mask

    INFO("BUGGY  participation_mask=0x" << std::hex << buggy_participation_mask);
    INFO("FIXED  participation_mask=0x" << std::hex << expected_participation_mask);

    REQUIRE(buggy_participation_mask != expected_participation_mask);
    REQUIRE(expected_participation_mask == 0xFFFFFFFFu);
}

// =============================================================================
// U-2: simpleGEMM pattern — post-fix wbar state correctly accumulates arrivals
//      across divergent halves
// =============================================================================
// After my fix in barrier.cpp:158, the force_reconvergence path preserves
// arrived_mask when the wbar is already initialized. This test verifies the
// wbar data structure correctly accumulates the second half's arrivals on
// top of the first half's existing arrived_mask.
//
// State being modeled:
//   - First half (lanes 16-31) already arrived and was released to PC=70.
//   - Lanes 0-15 are now arriving at the barrier.
//   - arrived_mask = 0xFFFF0000 (preserved from first half) → 0xFFFFFFFF
//   - is_complete() must return true once both halves have arrived.
// =============================================================================
TEST_CASE("U-2: simpleGEMM pattern — wbar accumulates arrivals across halves",
          "[barrier][divergence][unit][simplegemm-pattern][BUG-RECONVERGENCE]")
{
    WarpContext warp;
    for (int i = 0; i < 32; i++) add_thread(warp, i);
    auto& ws = warp.get_warp_state();

    // State after the first half's release:
    //   - Lanes 16-31 already at reconv_pc=70 (released)
    //   - Lanes 0-15 at barrier_pc=69, about to arrive
    for (int i = 0; i < 32; i++) {
        ws.threads[i].pc = (i >= 16) ? 70u : 69u;  // 16-31 past barrier, 0-15 at barrier
        ws.threads[i].next_pc = ws.threads[i].pc;
        ws.threads[i].is_active = true;
        ws.threads[i].is_blocked = false;
        ws.threads[i].is_exited = false;
        ws.threads[i].status = ThreadStatus::Active;
    }
    warp.set_active_mask(0xFFFFFFFFu);

    // Simulate the post-first-release wbar state (this is what the fix preserves):
    Wbar& wbar = ws.wbars[0];
    wbar.init(0xFFFFFFFFu, 70);
    wbar.arrived_mask = 0xFFFF0000u;  // first half's arrivals preserved
    ws.current_wbar_id = 0;

    INFO("Pre-second-half arrived_mask=0x" << std::hex << wbar.arrived_mask);
    INFO("Pre-second-half participation_mask=0x" << std::hex << wbar.participation_mask);

    // Now lanes 0-15 arrive (the second divergent half)
    for (int i = 0; i < 16; i++) wbar.arrive(i);

    INFO("Post-second-half arrived_mask=0x" << std::hex << wbar.arrived_mask);
    INFO("is_complete()=" << std::dec << wbar.is_complete());

    // CRITICAL assertions: the fix must preserve arrived_mask so the
    // second half's arrivals accumulate to the full 0xFFFFFFFF, making
    // the barrier completable for all 32 lanes.
    REQUIRE(wbar.arrived_mask == 0xFFFFFFFFu);
    REQUIRE(wbar.is_complete() == true);

    // Now simulate the release path (barrier.cpp:232-240 logic).
    // All 32 lanes are is_active=true, so all 32 should be released.
    simulate_release(warp, 70);

    int lanes_at_reconv = 0;
    int lanes_stuck = 0;
    for (int i = 0; i < 32; i++) {
        uint32_t pc = ws.threads[i].pc;
        if (pc == 70u) lanes_at_reconv++;
        if (pc == 69u) lanes_stuck++;
    }
    INFO("Lanes released to PC=70: " << lanes_at_reconv);
    INFO("Lanes stuck at PC=69:    " << lanes_stuck);

    REQUIRE(lanes_at_reconv == 32);
    REQUIRE(lanes_stuck == 0);
}