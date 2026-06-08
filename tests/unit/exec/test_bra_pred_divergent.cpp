// test_bra_pred_divergent.cpp
// =============================================================================
// Unit regression test for BUG-BRAPRED: bra_pred divergent path PC rewrite
// issue in warp_context.cpp:handle_branch (suspected SIMT stack return_pc issue
// at line 57).
//
// RED PHASE: This test must FAIL on unpatched code.
//
// Bug description:
//   In WarpContext::handle_branch (warp_context.cpp:57):
//     entry.return_pc = reconvergence_pc;
//   The SIMT stack entry's return_pc is set to reconvergence_pc, but this
//   might not correctly handle the divergent path PC rewrite. The suspected
//   issue is that:
//   1. return_pc should be fallthrough_pc (PC where not-taken lanes continue)
//   2. Or the PC rewrite for divergent paths doesn't correctly set the
//      fallthrough path's PC
//
// Expected behavior after fix:
//   When a divergent branch occurs:
//   - Taken lanes should have PC = target_pc
//   - Not-taken lanes should have PC = fallthrough_pc (current_inst_pc + 1)
//   - SIMT stack should correctly track the divergence
//   - Reconvergence should restore exec_mask correctly
//
// Test strategy:
//   1. Create a divergent branch scenario with @%p1 bra L_target
//   2. Set predicate so half the lanes take the branch, half don't
//   3. Verify that taken lanes have PC = target_pc
//   4. Verify that not-taken lanes have PC = fallthrough_pc
//   5. Verify SIMT stack depth is 1 after divergence
//   6. Verify exec_mask is taken_mask after divergence
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/simt_stack.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/memory_test_utils.h"
#include "ptxsim/testing/predicates.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"

#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

using ptxsim::testing::init_instruction_factory_once;
using ptxsim::testing::make_bra;
using ptxsim::testing::make_bra_pred;
using ptxsim::testing::make_mov;
using ptxsim::testing::make_mov_imm;
using ptxsim::testing::make_nop;
using ptxsim::testing::make_ret;
using ptxsim::testing::setup_block;
using ptxsim::testing::setup_pred;
using ptxsim::testing::step_warp;

// Test constants
static constexpr int BRANCH_PC = 4;
static constexpr int TARGET_PC = 20;
static constexpr int CONV_PC = 30;
static constexpr int FALLTHROUGH_PC = 5; // BRANCH_PC + 1
static constexpr int NUM_STMTS = 35;

// Helper to build instruction sequence
static std::vector<StatementContext> build_divergent_sequence(
    std::map<std::string, int> &l2pc) {
    std::vector<StatementContext> stmts;
    stmts.reserve(NUM_STMTS);

    // Fill with NOPs
    for (int i = 0; i < NUM_STMTS; i++) {
        stmts.push_back(make_nop());
    }

    // PC=4: @%p1 bra L_target (divergent branch)
    stmts[BRANCH_PC] = make_bra_pred("L_target", "%p1", false, CONV_PC);

    // PC=20: L_target (target of branch)
    // PC=30: L_reconv (reconvergence point)
    // PC=34: ret

    stmts[34] = make_ret();

    l2pc["L_target"] = TARGET_PC;
    l2pc["L_reconv"] = CONV_PC;

    return stmts;
}

// Helper to setup warp with divergence
static WarpContext *setup_divergent_warp(SMContext &sm,
                                         std::vector<StatementContext> &stmts,
                                         std::map<std::string, int> &l2pc,
                                         uint32_t taken_mask) {
    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1}, b{32, 1, 1}, bi{0, 0, 0};
    std::map<std::string, Symtable *> n2s;
    blk->init(g, b, bi, stmts, &n2s, l2pc);
    blk->sharedMemBytes = 1024;
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    WarpContext *w = sm.get_warp(0);
    REQUIRE(w != nullptr);

    // Setup predicate for divergence
    setup_pred(w, taken_mask);

    return w;
}

TEST_CASE("BUG-BRAPRED: divergent branch sets correct PC for taken and not-taken lanes",
          "[unit][regression][BUG-BRAPRED]") {
    // RED PHASE: This test must FAIL on unpatched code.
    // Bug: handle_branch may incorrectly set PC for divergent paths.
    // Expected: taken lanes → target_pc, not-taken lanes → fallthrough_pc

    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::map<std::string, int> l2pc;
    auto stmts = build_divergent_sequence(l2pc);

    SMContext sm(4, 128, 4096, 0);

    // Setup: lanes 0-15 take branch, lanes 16-31 don't
    uint32_t taken_mask = 0x0000FFFFu; // low 16 lanes take
    WarpContext *w = setup_divergent_warp(sm, stmts, l2pc, taken_mask);

    // Execute up to branch instruction
    for (int i = 0; i < BRANCH_PC; i++) {
        int pc = step_warp(w, stmts);
        CHECK(pc == i);
    }

    // Execute the branch instruction (PC=4)
    int pc = step_warp(w, stmts);
    CHECK(pc == BRANCH_PC);

    // After divergence:
    // 1. SIMT stack should have depth 1
    REQUIRE(w->get_simt_stack().depth() == 1);

    // 2. exec_mask should be taken_mask
    CHECK(w->get_exec_mask() == taken_mask);

    // 3. BUG CHECK: Verify PC for taken lanes (should be TARGET_PC)
    for (int lane = 0; lane < 16; lane++) {
        int lane_pc = w->get_thread_pc(lane);
        INFO("Lane " << lane << " PC after divergence: " << lane_pc
             << " (expected " << TARGET_PC << ")");
        CHECK(lane_pc == TARGET_PC);
    }

    // 4. BUG CHECK: Verify PC for not-taken lanes (should be FALLTHROUGH_PC)
    for (int lane = 16; lane < 32; lane++) {
        int lane_pc = w->get_thread_pc(lane);
        INFO("Lane " << lane << " PC after divergence: " << lane_pc
             << " (expected " << FALLTHROUGH_PC << ")");
        CHECK(lane_pc == FALLTHROUGH_PC);
    }

    // 5. Verify SIMT stack entry contents
    auto &entry = w->get_simt_stack().top();
    CHECK(entry.branch_pc == BRANCH_PC);
    CHECK(entry.reconvergence_pc == CONV_PC);
    CHECK(entry.active_mask == taken_mask);
    CHECK(entry.return_mask == 0xFFFFFFFFu); // original exec_mask

    // BUG CHECK: return_pc should be set correctly
    // The suspected bug is that return_pc might be incorrectly set
    INFO("SIMT stack entry.return_pc = " << entry.return_pc
         << " (reconvergence_pc = " << CONV_PC << ")");
}

TEST_CASE("BUG-BRAPRED: divergent branch with inverted predicate",
          "[unit][regression][BUG-BRAPRED]") {
    // RED PHASE: This test must FAIL on unpatched code.
    // Extended test: verify divergence with @!%p1 bra (negated predicate)

    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::map<std::string, int> l2pc;
    std::vector<StatementContext> stmts;
    stmts.reserve(NUM_STMTS);

    for (int i = 0; i < NUM_STMTS; i++) {
        stmts.push_back(make_nop());
    }

    // PC=4: @!%p1 bra L_target (negated predicate - lanes with p1=0 take)
    stmts[BRANCH_PC] = make_bra_pred("L_target", "%p1", true, CONV_PC); // neg=true
    stmts[34] = make_ret();

    l2pc["L_target"] = TARGET_PC;
    l2pc["L_reconv"] = CONV_PC;

    SMContext sm(4, 128, 4096, 0);

    // Setup: lanes 0-15 have p1=1, lanes 16-31 have p1=0
    // With negated predicate, lanes 16-31 take the branch
    uint32_t p1_true_mask = 0x0000FFFFu;
    WarpContext *w = setup_divergent_warp(sm, stmts, l2pc, p1_true_mask);

    // Expected: lanes 16-31 take (p1=0, negated), lanes 0-15 don't take
    uint32_t expected_taken_mask = 0xFFFF0000u; // high 16 lanes

    // Execute up to branch
    for (int i = 0; i < BRANCH_PC; i++) {
        step_warp(w, stmts);
    }

    // Execute branch
    step_warp(w, stmts);

    // BUG CHECK: Verify divergence occurred
    REQUIRE(w->get_simt_stack().depth() == 1);

    // exec_mask should be expected_taken_mask (high 16 lanes)
    CHECK(w->get_exec_mask() == expected_taken_mask);

    // BUG CHECK: Verify PC for taken lanes (high 16, should be TARGET_PC)
    for (int lane = 16; lane < 32; lane++) {
        int lane_pc = w->get_thread_pc(lane);
        INFO("Lane " << lane << " (taken) PC: " << lane_pc
             << " (expected " << TARGET_PC << ")");
        CHECK(lane_pc == TARGET_PC);
    }

    // BUG CHECK: Verify PC for not-taken lanes (low 16, should be FALLTHROUGH_PC)
    for (int lane = 0; lane < 16; lane++) {
        int lane_pc = w->get_thread_pc(lane);
        INFO("Lane " << lane << " (not-taken) PC: " << lane_pc
             << " (expected " << FALLTHROUGH_PC << ")");
        CHECK(lane_pc == FALLTHROUGH_PC);
    }
}

TEST_CASE("BUG-BRAPRED: SIMT stack return_pc consistency",
          "[unit][regression][BUG-BRAPRED]") {
    // RED PHASE: This test must FAIL on unpatched code.
    // Test: Verify that SIMT stack entry.return_pc is consistent with
    // reconvergence behavior.

    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::map<std::string, int> l2pc;
    auto stmts = build_divergent_sequence(l2pc);

    SMContext sm(4, 128, 4096, 0);
    uint32_t taken_mask = 0x0000FFFFu;
    WarpContext *w = setup_divergent_warp(sm, stmts, l2pc, taken_mask);

    // Execute to divergence point
    for (int i = 0; i <= BRANCH_PC; i++) {
        step_warp(w, stmts);
    }

    REQUIRE(w->get_simt_stack().depth() == 1);

    auto &entry = w->get_simt_stack().top();

    // BUG CHECK: return_pc should equal reconvergence_pc
    // If this is incorrect, reconvergence won't work properly
    CHECK(entry.return_pc == entry.reconvergence_pc);

    // Additional check: return_pc should be CONV_PC
    CHECK(entry.return_pc == CONV_PC);

    INFO("entry.return_pc = " << entry.return_pc
         << ", entry.reconvergence_pc = " << entry.reconvergence_pc);
}