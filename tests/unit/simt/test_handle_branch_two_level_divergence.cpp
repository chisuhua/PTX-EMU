/**
 * Unit test: handle_branch convergence with pre-existing secondary entry on SIMT stack.
 *
 * Migrated from tests/integration/divergence/test_divergence_sync_convergence.cpp
 * (former Test B). The test manually pushes a secondary SIMTStackEntry to simulate the
 * state that would normally be created by a real loop's back-edge. Per the project's
 * zero-tolerance policy on integration tests, this test belongs in unit/ where direct
 * SIMT stack manipulation is allowed.
 *
 * Verifies that handle_branch correctly handles two-level divergence: when a secondary
 * entry is already on the stack (from a back-edge), the scheduler must:
 *   1. Drive Path A from PC 5 to PC 13
 *   2. At CONV_PC, pop the secondary entry when its active_mask converges
 *   3. Block Path A lanes that are not in the primary entry's active_mask
 *   4. Switch to Path B (PC 28) and execute it
 *   5. Pop the primary entry and unblock Path A
 */
#include "catch_amalgamated.hpp"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/simt_stack.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/predicates.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
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

static constexpr int      BRANCH_PC     = 4;
static constexpr int      CONV_PC       = 14;
static constexpr int      PATH_A_START  = 5;
static constexpr int      PATH_A_END    = 13;
static constexpr int      PATH_B_TARGET = 28;
static constexpr int      PATH_B_END    = 33;
static constexpr int      BRA_UNI_PC    = 34;
static constexpr int      NUM_STMTS     = 35;

static void init_factory() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        ptxsim::DebugConfig::get().set_trace_simt_stack_enabled(true);
        ptxsim::DebugConfig::get().set_trace_divergence_enabled(true);
        ptxsim::DebugConfig::get().set_trace_convergence_enabled(true);
        ptxsim::LoggerConfig::get().set_component_level("emu", ptxsim::log_level::debug);
        done = true;
    }
}

static std::vector<StatementContext> build_instrs(
    std::map<std::string, int> &l2pc)
{
    std::vector<StatementContext> v;
    v.reserve(NUM_STMTS);
    for (int i = 0; i < NUM_STMTS; i++) v.push_back(ptxsim::testing::make_nop());
    v[BRANCH_PC] = ptxsim::testing::make_bra_pred("L__BB0_4", "%p1", false, CONV_PC);
    v[BRA_UNI_PC] = ptxsim::testing::make_bra("L__BB0_3");
    v[27] = ptxsim::testing::make_ret();
    l2pc["L__BB0_4"] = PATH_B_TARGET;
    l2pc["L__BB0_3"] = CONV_PC;
    return v;
}

static WarpContext* setup(SMContext &sm,
                          std::vector<StatementContext> &v,
                          std::map<std::string, int> &l2pc)
{
    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1,1,1}, b{32,1,1}, bi{0,0,0};
    std::map<std::string, std::unique_ptr<Symtable>> n2s;
    blk->init(g, b, bi, v, &n2s, l2pc);
    blk->sharedMemBytes = 1024;
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok); return sm.get_warp(0);
}

TEST_CASE("handle_branch: two level div with pre-existing secondary entry",
          "[simt][handle_branch][divergence][two_level]")
{
    init_factory(); ResourceManager::instance().initialize(1, 8192);
    std::map<std::string, int> l2pc;
    auto v = build_instrs(l2pc);
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup(sm, v, l2pc);
    setup_pred(w, 0x0000FFFFu);

    step_warp(w, v); step_warp(w, v);
    step_warp(w, v); step_warp(w, v);
    step_warp(w, v);  // @%p1 bra → 分歧

    // Manually push secondary entry to simulate loop back-edge state.
    // (This is allowed in unit/ — tests handle_branch's ability to process
    // a pre-existing secondary entry.)
    ptxsim::SIMTStackEntry le;
    le.branch_pc = 13; le.reconvergence_pc = CONV_PC;
    le.active_mask = 0xFFFC0000u; // lanes 18-31
    le.return_mask = 0xFFFF0000u; le.return_pc = CONV_PC;
    w->get_simt_stack().push(le);
    REQUIRE(w->get_simt_stack().depth() == 2);

    // step_warp drives Path A (PC 5-13)
    int pc;
    pc = step_warp(w, v); CHECK(pc == PATH_A_START);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 1);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 2);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 3);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 4);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 5);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 6);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 7);
    pc = step_warp(w, v); CHECK(pc == PATH_A_END);

    // Path A arrives at PC=14 → secondary entry's active_mask (18-31) converges → pop
    CHECK(w->get_simt_stack().depth() == 1);

    // Primary entry top: Path A blocked at PC 14 (lanes not in active_mask 0x0000FFFF)
    pc = step_warp(w, v);
    CHECK(pc == CONV_PC);
    CHECK(w->get_warp_state().threads[16].is_blocked == true);

    // Scheduler switches to Path B (PC 28)
    CHECK(step_warp(w, v) == PATH_B_TARGET);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 1);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 2);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 3);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 4);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 5);
    CHECK(step_warp(w, v) == BRA_UNI_PC);

    // Reconvergence: primary entry popped, Path A unblocked
    CHECK(w->get_simt_stack().empty());
    CHECK(w->get_exec_mask() == 0xFFFFFFFFu);
    CHECK(w->get_warp_state().threads[16].is_blocked == false);
}
