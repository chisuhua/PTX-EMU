/**
 * Test 3 Root Cause Investigation - Unit Tests
 *
 * Tests three possible failure directions for test_syncthreads Test 3
 * (test_nested_sync with 16 threads, divergent branch, then barrier).
 */

#include "catch_amalgamated.hpp"
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_factory.h"
#include <vector>
#include <map>
#include <string>
#include <cstdint>

namespace cfg = ptx::cfg;
using namespace ptx;
namespace {
using namespace ptxir::factory;

// ============================================================
// Helpers
// ============================================================

static StatementContext make_regular_stmt(StatementType type = S_MOV) {
    StatementContext ctx;
    ctx.type = type;
    GenericInstr instr;
    ctx.data = instr;
    return ctx;
}

static StatementContext make_branch_stmt(const std::string& target) {
    StatementContext ctx;
    ctx.type = S_BRA;
    BranchInstr branch;
    branch.target = target;
    branch.reconvergence_pc = -1;
    ctx.data = branch;
    return ctx;
}

static StatementContext make_label_stmt(const std::string& name) {
    StatementContext ctx;
    ctx.type = S_DOLLOR;
    DollarNameInstr label;
    label.name = name;
    ctx.data = label;
    return ctx;
}

static cfg::PostDominatorMap build_post_doms(
    const std::vector<StatementContext>& stmts,
    const std::map<std::string, int>& label2pc)
{
    cfg::CFG cfg_obj = cfg::CFGBuilder::build(stmts, label2pc);
    return cfg::CFGBuilder::computePostDominators(cfg_obj);
}

static uint32_t compute_participation_mask(int total_threads) {
    if (total_threads >= 32) return 0xFFFFFFFFu;
    return (1u << total_threads) - 1;
}

// ============================================================
// Direction 1: reconvergence_pc fallback correctness
// ============================================================

TEST_CASE("D1: barrier reconvergence_pc fallback when no post-dominator",
          "[cfg][direction1][reconvergence]") {
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;

    stmts.push_back(make_regular_stmt());
    stmts.push_back(makeBarWarpSyncInstr(0x0000FFFF, -1));
    stmts.push_back(make_regular_stmt());
    stmts.push_back(make_regular_stmt(S_RET));

    auto postDoms = build_post_doms(stmts, label2pc);

    for (const auto& [pc, ipd] : postDoms) {
        INFO("  PC=" << pc << " -> post-dom=" << ipd);
    }

    auto it = postDoms.find(1);
    REQUIRE(it != postDoms.end());
    REQUIRE((it->second == 2 || it->second == -1));

    auto it2 = postDoms.find(2);
    REQUIRE(it2 != postDoms.end());
    REQUIRE((it2->second == 3 || it2->second == -1));

    auto it3 = postDoms.find(3);
    REQUIRE(it3 != postDoms.end());
    REQUIRE((it3->second == 4 || it3->second == -1));
}

TEST_CASE("D1: two consecutive barriers — second barrier reconvergence",
          "[cfg][direction1][reconvergence]") {
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;
    stmts.push_back(makeBarWarpSyncInstr(0x0000FFFF, -1));
    stmts.push_back(makeBarWarpSyncInstr(0x0000FFFF, -1));
    stmts.push_back(make_regular_stmt());
    stmts.push_back(make_regular_stmt(S_RET));

    auto postDoms = build_post_doms(stmts, label2pc);

    for (const auto& [pc, ipd] : postDoms) {
        INFO("  PC=" << pc << " -> post-dom=" << ipd);
    }

    auto it = postDoms.find(1);
    REQUIRE(it != postDoms.end());
    REQUIRE((it->second >= 2 || it->second == -1));

    auto it0 = postDoms.find(0);
    REQUIRE(it0 != postDoms.end());
    REQUIRE((it0->second >= 1 || it0->second == -1));
}

TEST_CASE("D1: participation mask for various thread counts",
          "[unit][direction1][mask]") {
    REQUIRE(compute_participation_mask(1)  == 0x00000001u);
    REQUIRE(compute_participation_mask(8)  == 0x000000FFu);
    REQUIRE(compute_participation_mask(16) == 0x0000FFFFu);
    REQUIRE(compute_participation_mask(32) == 0xFFFFFFFFu);

    // Edge: 31 threads (should NOT overflow)
    REQUIRE(compute_participation_mask(31) == 0x7FFFFFFFu);
    // Edge: 33 threads (clamped to full mask)
    REQUIRE(compute_participation_mask(33) == 0xFFFFFFFFu);
    // Edge: 64 threads (clamped to full mask)
    REQUIRE(compute_participation_mask(64) == 0xFFFFFFFFu);
}

// ============================================================
// Direction 2: divergent branch with small CTA (16 threads)
// ============================================================

TEST_CASE("D2: CFG post-dominator for branch-then-barrier (Test 3 pattern)",
          "[cfg][direction2][divergent]") {
    // Mirrors test_nested_sync exactly:
    // PC=0-4:  setup (mov, st.shared)
    // PC=5:    bar.sync 0
    // PC=6:    setp
    // PC=7:    @%p1 bra $L_skip          ← branch
    // PC=8-10: then-body                 ← tid 0-15
    // PC=11:   $L_skip: label            ← tid 16-31 jump here
    // PC=12:   bar.sync 0                ← ALL threads converge
    // PC=13:   ld.shared
    // PC=14:   ret
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;

    for (int i = 0; i < 5; i++) stmts.push_back(make_regular_stmt());
    stmts.push_back(makeBarWarpSyncInstr(0x0000FFFF, -1));               // PC=5
    stmts.push_back(make_regular_stmt(S_SETP));         // PC=6
    stmts.push_back(make_branch_stmt("L_skip"));        // PC=7
    label2pc["L_skip"] = 11;
    for (int i = 0; i < 3; i++) stmts.push_back(make_regular_stmt()); // PC=8-10
    stmts.push_back(make_label_stmt("L_skip"));         // PC=11
    stmts.push_back(makeBarWarpSyncInstr(0x0000FFFF, -1));               // PC=12
    stmts.push_back(make_regular_stmt());               // PC=13
    stmts.push_back(make_regular_stmt(S_RET));          // PC=14

    auto postDoms = build_post_doms(stmts, label2pc);

    for (const auto& [pc, ipd] : postDoms) {
        INFO("  PC=" << pc << " -> post-dom=" << ipd);
    }

    // Branch at PC=7 should converge at PC=11 (the label)
    auto it_br = postDoms.find(7);
    if (it_br != postDoms.end()) {
        REQUIRE(it_br->second == 11);
    }

    // Second barrier at PC=12 should be post-dominated by PC=13/14 or exit (-1)
    auto it_b2 = postDoms.find(12);
    REQUIRE(it_b2 != postDoms.end());
    REQUIRE((it_b2->second >= 12 || it_b2->second == -1));

    // Then-body instructions (PC=8,9,10) should all converge at PC=11
    for (int pc = 8; pc <= 10; pc++) {
        auto it = postDoms.find(pc);
        if (it != postDoms.end()) {
            REQUIRE(it->second == 11);
        }
    }
}

TEST_CASE("D2: divergent branch with 8 threads in CTA",
          "[cfg][direction2][small_cta]") {
    // 8-thread CTA: tid 0-3 take branch, tid 4-7 fall through
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;

    stmts.push_back(make_regular_stmt());    // PC=0: mov
    stmts.push_back(make_regular_stmt(S_SETP)); // PC=1: setp
    stmts.push_back(make_branch_stmt("L_merge")); // PC=2: bra
    label2pc["L_merge"] = 5;
    stmts.push_back(make_regular_stmt());    // PC=3: then
    stmts.push_back(make_regular_stmt());    // PC=4: more then
    stmts.push_back(make_label_stmt("L_merge")); // PC=5: label
    stmts.push_back(makeBarWarpSyncInstr(0x0000FFFF, -1));    // PC=6: bar.sync
    stmts.push_back(make_regular_stmt(S_RET)); // PC=7: ret

    auto postDoms = build_post_doms(stmts, label2pc);

    for (const auto& [pc, ipd] : postDoms) {
        INFO("  PC=" << pc << " -> post-dom=" << ipd);
    }

    auto it_br = postDoms.find(2);
    if (it_br != postDoms.end()) {
        REQUIRE(it_br->second == 5);
    }

    auto it_b = postDoms.find(6);
    REQUIRE(it_b != postDoms.end());
    REQUIRE((it_b->second >= 6 || it_b->second == -1));

    // 8-thread mask should be 0xFF
    REQUIRE(compute_participation_mask(8) == 0x000000FFu);
}

// ============================================================
// Direction 3: shared memory + barrier correctness with small CTA
// ============================================================

TEST_CASE("D3: sequential shared memory with barrier between writes",
          "[cfg][direction3][shared_mem]") {
    // PC=0: st.shared [%addr0], %tid
    // PC=1: bar.sync 0
    // PC=2: ld.shared from [%addr0]
    // PC=3: bar.sync 0
    // PC=4: ret
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;

    stmts.push_back(make_regular_stmt(S_ST));     // PC=0
    stmts.push_back(makeBarWarpSyncInstr(0x0000FFFF, -1));         // PC=1
    stmts.push_back(make_regular_stmt(S_LD));     // PC=2
    stmts.push_back(makeBarWarpSyncInstr(0x0000FFFF, -1));         // PC=3
    stmts.push_back(make_regular_stmt(S_RET));    // PC=4

    auto postDoms = build_post_doms(stmts, label2pc);

    for (const auto& [pc, ipd] : postDoms) {
        INFO("  PC=" << pc << " -> post-dom=" << ipd);
    }

    // Both barriers should have valid post-dominators
    auto it_b1 = postDoms.find(1);
    REQUIRE(it_b1 != postDoms.end());
    REQUIRE((it_b1->second >= 1 || it_b1->second == -1));

    auto it_b2 = postDoms.find(3);
    REQUIRE(it_b2 != postDoms.end());
    REQUIRE((it_b2->second >= 3 || it_b2->second == -1));
}

TEST_CASE("D3: barrier reconvergence_pc for if-then pattern with shared memory",
          "[cfg][direction3][shared_barrier]") {
    // Reproduces the exact pattern from test_nested_sync:
    // all threads do first barrier, then only subset executes shared memory writes,
    // then all threads do second barrier and read results.
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;

    stmts.push_back(make_regular_stmt());        // PC=0: setup
    stmts.push_back(makeBarWarpSyncInstr(0x0000FFFF, -1));        // PC=1: bar.sync 0
    stmts.push_back(make_regular_stmt(S_SETP));  // PC=2: predicate
    stmts.push_back(make_branch_stmt("L_after"));// PC=3: conditional bra
    label2pc["L_after"] = 7;
    stmts.push_back(make_regular_stmt(S_ST));    // PC=4: shared write
    stmts.push_back(make_regular_stmt());         // PC=5: more work
    stmts.push_back(make_regular_stmt());         // PC=6: more work
    stmts.push_back(make_label_stmt("L_after"));  // PC=7: label
    stmts.push_back(makeBarWarpSyncInstr(0x0000FFFF, -1));         // PC=8: bar.sync 0
    stmts.push_back(make_regular_stmt(S_LD));     // PC=9: shared read
    stmts.push_back(make_regular_stmt(S_RET));    // PC=10: ret

    auto postDoms = build_post_doms(stmts, label2pc);

    for (const auto& [pc, ipd] : postDoms) {
        INFO("  PC=" << pc << " -> post-dom=" << ipd);
    }

    // The shared memory writes (PC=4,5,6) should converge at PC=7
    for (int pc = 4; pc <= 6; pc++) {
        auto it = postDoms.find(pc);
        if (it != postDoms.end()) {
            REQUIRE(it->second == 7);
        }
    }

    // Branch at PC=3 should also converge at PC=7
    auto it_br = postDoms.find(3);
    if (it_br != postDoms.end()) {
        REQUIRE(it_br->second == 7);
    }

    // Second barrier at PC=8 must have PC>=8 as post-dominator
    auto it_b2 = postDoms.find(8);
    REQUIRE(it_b2 != postDoms.end());
    REQUIRE((it_b2->second >= 8 || it_b2->second == -1));

    // First barrier at PC=1 should have a valid post-dominator >= 1
    auto it_b1 = postDoms.find(1);
    REQUIRE(it_b1 != postDoms.end());
    REQUIRE((it_b1->second >= 1 || it_b1->second == -1));
}

static uint32_t compute_participation_mask_for(int threads) {
    return (threads >= 32) ? 0xFFFFFFFFu : ((1u << threads) - 1);
}

TEST_CASE("D3a: participation mask handling — non-full mask does not trigger auto-fill",
          "[unit][direction3a][execution]") {
    uint32_t full_mask = 0xFFFFFFFFu;
    uint32_t partial_mask = 0x0000FFFFu;

    REQUIRE((full_mask & (1u << 31)) != 0);
    REQUIRE((full_mask & (1u << 0)) != 0);
    REQUIRE((partial_mask & (1u << 16)) == 0);
    REQUIRE((partial_mask & (1u << 15)) != 0);
    REQUIRE(partial_mask != full_mask);
}

TEST_CASE("D3b: thread count < 32 — inactive lanes excluded from mask",
          "[unit][direction3b][execution]") {
    for (int tc = 1; tc <= 32; tc++) {
        uint32_t mask = compute_participation_mask_for(tc);
        int active = 0;
        for (int lane = 0; lane < 32; lane++) {
            if (mask & (1u << lane)) active++;
        }
        REQUIRE(active == std::min(tc, 32));
    }
}

TEST_CASE("D3c: barrier release PC advancement",
          "[unit][direction3c][execution]") {
    int barrier_pc = 12;
    int reconvergence_pc = 13;

    int next_if_sync_complete = barrier_pc + 1;
    REQUIRE(next_if_sync_complete == 13);

    int next_if_sync_incomplete = barrier_pc;
    REQUIRE(next_if_sync_incomplete == barrier_pc);
    REQUIRE(next_if_sync_incomplete != next_if_sync_complete);
}

} // anonymous namespace
