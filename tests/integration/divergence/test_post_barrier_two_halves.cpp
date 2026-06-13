/**
 * 指令序列集成测试：验证 BUG-POSTBARRIER-TWOHALVES
 *
 * Bug: When the barrier handler is invoked via the force_reconvergence path
 * (unique_pcs.size() > 1), it initializes wbar with only the currently
 * arriving half. The second call overwrites active_mask with only the
 * second half, losing lanes already released by the first call.
 *
 * Reproduction:
 *   1. Two divergent paths. Path A is 1 instruction, Path B uses bra.uni
 *      from a higher PC to jump back to the barrier.
 *   2. Path A reaches barrier first → force_reconvergence → active_mask = FFFF0000
 *   3. Path B bra.uni jumps to barrier → second force_reconvergence →
 *      active_mask = 0000FFFF (BUG: loses lanes 16-31)
 *   4. update_active_mask self-heals in the next step, so the final warp
 *      state may look correct. The bug manifests as active_mask being
 *      temporarily wrong.
 */
#include "catch_amalgamated.hpp"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/simt_stack.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/predicates.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"
#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <cstdint>

using ptxsim::testing::step_warp;
using ptxsim::testing::setup_pred;
using ptxsim::testing::make_nop;
using ptxsim::testing::make_bra_pred;
using ptxsim::testing::make_bra;
using ptxsim::testing::make_bar_warp_sync;
using ptxsim::testing::make_ret;

static constexpr int BRANCH_PC    = 1;
static constexpr int PATH_A_PC    = 2;
static constexpr int BARRIER_PC   = 3;
static constexpr int PATH_B_PC    = 4;
static constexpr int POST_BARRIER = 5;
static constexpr int RET_PC       = 6;
static constexpr int NUM_STMTS    = 7;

static void init_factory() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

static std::vector<StatementContext> build_instrs(
    std::map<std::string, int>& l2pc)
{
    std::vector<StatementContext> v(NUM_STMTS);
    for (auto& s : v) s = make_nop();
    v[BRANCH_PC]  = make_bra_pred("L_PATH_B", "%p1", false, BARRIER_PC);
    v[BARRIER_PC] = make_bar_warp_sync(0xFFFFFFFFu, POST_BARRIER);
    // Path B starts at PC=4 (higher than barrier PC=3) and jumps to barrier
    v[PATH_B_PC]  = make_bra("L_BARRIER");
    v[POST_BARRIER] = make_nop();
    v[RET_PC]     = make_ret();
    l2pc["L_PATH_B"]  = PATH_B_PC;
    l2pc["L_BARRIER"] = BARRIER_PC;
    return v;
}

static WarpContext* setup_warp(SMContext& sm,
                               std::vector<StatementContext>& v,
                               std::map<std::string, int>& l2pc)
{
    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1,1,1}, b{32,1,1}, bi{0,0,0};
    std::map<std::string, Symtable*> n2s;
    blk->init(g, b, bi, v, &n2s, l2pc);
    blk->sharedMemBytes = 1024;
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    return sm.get_warp(0);
}

// ============================================================================
// Test 1: After both halves pass the barrier, all 32 lanes must be past
// the barrier. With the bug, the second barrier release overwrites
// active_mask, and lanes 16-31 may be temporarily lost.
// ============================================================================
TEST_CASE("post-barrier: all 32 lanes advance past barrier (two halves)",
          "[barrier][divergence][integrated][regression][BUG-POSTBARRIER-TWOHALVES]")
{
    init_factory();
    ResourceManager::instance().initialize(1, 8192);

    std::map<std::string, int> l2pc;
    auto v = build_instrs(l2pc);
    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup_warp(sm, v, l2pc);

    // Lanes 0-15: p1=true → branch to Path B (PC=4)
    // Lanes 16-31: p1=false → fall through to Path A (PC=2)
    setup_pred(w, 0x0000FFFFu);

    // Step through the entire sequence
    constexpr int MAX_STEPS = 100;
    for (int i = 0; i < MAX_STEPS; i++) {
        if (w->is_finished()) break;
        step_warp(w, v);
    }

    // After all steps, verify all lanes passed the barrier.
    // With the bug, lanes 16-31 are stuck at the barrier (never released
    // because the second barrier release overwrote active_mask to 0000FFFF).
    int stuck_at_barrier = 0;
    int stuck_at_path_b = 0;
    for (int lane = 0; lane < 32; lane++) {
        uint32_t lane_pc = w->get_warp_state().threads[lane].pc;
        if (lane_pc == BARRIER_PC) stuck_at_barrier++;
        if (lane_pc == PATH_B_PC)  stuck_at_path_b++;
    }
    INFO("Lanes stuck at BARRIER_PC: " << stuck_at_barrier);
    INFO("Lanes stuck at PATH_B_PC: "  << stuck_at_path_b);
    CHECK(stuck_at_barrier == 0);
    CHECK(stuck_at_path_b  == 0);
}
