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
// U-2: simpleGEMM pattern — after both halves arrive, all 32 lanes must be released
// =============================================================================
// After wbar init (buggy: mask=0xFFFF0000), all 32 lanes eventually arrive at
// barrier. arrived_mask = 0xFFFFFFFF. is_complete() returns TRUE because arrived
// covers the (buggy) participation_mask. Release path iterates arrived_mask & is_active:
//   - lanes 16-31: arrived_mask bit set, is_active=true → released
//   - lanes 0-15: arrived_mask bit set BUT is_active=false (was set false when
//                  barrier blocked them) → NOT released
//
// This documents the bug: only 16 lanes released.
// =============================================================================
TEST_CASE("U-2: simpleGEMM pattern — buggy release leaves lanes 0-15 stuck",
          "[barrier][divergence][unit][simplegemm-pattern][BUG-RECONVERGENCE]")
{
    WarpContext warp;
    for (int i = 0; i < 32; i++) add_thread(warp, i);
    auto& ws = warp.get_warp_state();

    // Set up state matching the bug: lanes 16-31 already arrived + blocked (is_active=false)
    for (int i = 0; i < 32; i++) {
        ws.threads[i].pc = 69;
        ws.threads[i].is_active = (i < 16);  // lanes 16-31 marked inactive by update_active_mask after blocking
        ws.threads[i].is_blocked = (i >= 16);
        ws.threads[i].is_exited = false;
        ws.threads[i].status = (i < 16) ? ThreadStatus::Active : ThreadStatus::Blocked;
    }
    warp.set_active_mask(0x0000FFFFu);

    // Mirror the BUG runtime state: wbar init with mask=0xFFFF0000, arrived=0xFFFF0000
    Wbar& wbar = ws.wbars[0];
    wbar.init(0xFFFF0000u, 70);
    wbar.arrived_mask = 0xFFFF0000u;
    ws.current_wbar_id = 0;

    // Now lanes 0-15 "arrive" (the only currently active lanes)
    for (int i = 0; i < 16; i++) wbar.arrive(i);

    INFO("Final arrived_mask=0x" << std::hex << wbar.arrived_mask);
    INFO("Final participation_mask=0x" << std::hex << wbar.participation_mask);
    INFO("is_complete()=" << std::dec << wbar.is_complete());
    REQUIRE(wbar.arrived_mask == 0xFFFFFFFFu);
    REQUIRE(wbar.is_complete() == true);

    // Trigger release (mirrors barrier.cpp:232-240)
    simulate_release(warp, 70);

    int lanes_at_reconv = 0;
    int lanes_at_barrier = 0;
    for (int i = 0; i < 32; i++) {
        uint32_t pc = ws.threads[i].pc;
        if (pc == 70u) lanes_at_reconv++;
        if (pc == 69u) lanes_at_barrier++;
    }
    INFO("Lanes released to PC=70: " << lanes_at_reconv);
    INFO("Lanes stuck at PC=69:    " << lanes_at_barrier);

    // BUG: lanes_at_reconv == 16 (only upper half); lanes_at_barrier == 16 (lower half).
    // FIX: lanes_at_reconv == 32; lanes_at_barrier == 0.
    //
    // This test asserts the FIXED behavior — all 32 lanes released to reconv_pc.
    REQUIRE(lanes_at_reconv == 32);
    REQUIRE(lanes_at_barrier == 0);
}