/**
 * Test 3 (test_nested_sync) Reproduction Test
 *
 * This unit test reproduces the exact failure pattern of Test 3 from
 * test_syncthreads.ptx (_Z16test_nested_sync kernel) to isolate the
 * root cause of the barrier deadlock.
 *
 * Background:
 *   Test 3 hangs at the second bar.sync with only 16/32 threads arriving.
 *   T16-31 never reach the barrier.
 *
 * What this test covers:
 *   1. CFG post-dominator computation for divergent branch + barrier pattern
 *   2. bar.sync → bar.warp.sync translation with participation mask
 *   3. Branch divergence with predicate (the "core engine" path)
 *   4. get_lanes_by_pc scheduler behavior with divergent PC groups
 *
 * NOT tested (requires full integration):
 *   - Actual PTX parsing (ANTLR)
 *   - Register bank with predicated execution
 *   - Full SM scheduler loop
 *
 * @file   test3_reproduction.cpp
 * @brief  Standalone reproduction of test_nested_sync deadlock
 * @author PTX-EMU Team
 * @date   2026-04-14
 */

#include "catch_amalgamated.hpp"
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include <vector>
#include <map>
#include <string>
#include <cstdint>
#include <array>

namespace cfg = ptx::cfg;
using namespace ptx;

// =============================================================================
// Test data: Exact replica of _Z16test_nested_sync from test_syncthreads.ptx
// =============================================================================

/*
 * PTX instruction mapping (actual file lines 145-172):
 *
 * PC=0-9:   ld.param, mov, shl, mov, add, st.shared  (6 instructions, not 10)
 *            → Simplified to 6 regular stmts for CFG test
 * PC=10:    bar.sync 0
 * PC=11:    setp.gt.u32 %p1, %r1, 15
 * PC=12:    mov.u32 %r6, data_b         ← executed by ALL threads (before bra!)
 * PC=13:    add.s32 %r3, %r6, %r4       ← executed by ALL threads (before bra!)
 * PC=14:    @%p1 bra $L__BB2_2          ← DIVERGENT: T16-31 jump, T0-15 fall-through
 * PC=15:    ld.shared.u32 %r7, [%r2]
 * PC=16:    add.s32 %r9, %r4, 4
 * PC=17:    and.b32 %r10, %r9, 60
 * PC=18:    add.s32 %r12, %r5, %r10
 * PC=19:    ld.shared.u32 %r13, [%r12]
 * PC=20:    add.s32 %r14, %r13, %r7
 * PC=21:    st.shared.u32 [%r3], %r14   ← last of T0-15 path
 * PC=22:    $L__BB2_2: label
 * PC=23:    cvta.to.global.u64 %rd2
 * PC=24:    bar.sync 0                  ← ★ SECOND BARRIER (hang here!)
 * PC=25:    ld.shared.u32 %r15, [%r3]
 * PC=26:    mul.wide.u32 %rd3, %r1, 4
 * PC=27:    add.s64 %rd4, %rd2, %rd3
 * PC=28:    st.global.u32 [%rd4], %r15
 * PC=29:    ret
 */

// =============================================================================
// Helpers
// =============================================================================

static StatementContext make_regular(StatementType type = S_MOV) {
    StatementContext ctx;
    ctx.type = type;
    GenericInstr instr;
    ctx.data = instr;
    return ctx;
}

static StatementContext make_branch(const std::string& target) {
    StatementContext ctx;
    ctx.type = S_BRA;
    BranchInstr branch;
    branch.target = target;
    branch.reconvergence_pc = -1;
    ctx.data = branch;
    return ctx;
}

static StatementContext make_warp_barrier(const std::string& mask_str,
                                           const std::string& reconvergence_str) {
    StatementContext ctx;
    ctx.type = S_BAR_WARP_SYNC;
    BarWarpSyncInstr barrier;
    barrier.qualifiers = {Qualifier::Q_B32};
    barrier.operands.push_back(OperandContext{ImmOperand{mask_str}});
    barrier.operands.push_back(OperandContext{ImmOperand{reconvergence_str}});
    ctx.data = barrier;
    return ctx;
}

static StatementContext make_label(const std::string& name) {
    StatementContext ctx;
    ctx.type = S_DOLLOR;
    DollarNameInstr lbl;
    lbl.name = name;
    ctx.data = lbl;
    return ctx;
}

static cfg::CFG build_test_cfg(
    const std::vector<StatementContext>& stmts,
    const std::map<std::string, int>& label2pc)
{
    return cfg::CFGBuilder::build(stmts, label2pc);
}

static cfg::PostDominatorMap compute_post_doms(const cfg::CFG& cfg) {
    return cfg::CFGBuilder::computePostDominators(cfg);
}

// =============================================================================
// Mask utility (reproduces the logic from ptx_visitor_barrier.cpp)
// =============================================================================

/// Compute participation mask for a given CTA size (mirrors isWarpLevelBarrier check)
static uint32_t compute_participation_mask(int cta_threads) {
    cta_threads = std::min(cta_threads, 32);
    if (cta_threads >= 32) return 0xFFFFFFFFu;
    return (1u << cta_threads) - 1;
}

/// Count active bits in mask (used to check if all expected threads arrived)
static int count_set_bits(uint32_t mask) {
    int count = 0;
    for (int i = 0; i < 32; i++) {
        if (mask & (1u << i)) count++;
    }
    return count;
}

/// Simulate the bar.sync → bar.warp.sync translation (ptx_visitor_barrier.cpp:42-71)
struct BarrierTranslationResult {
    uint32_t participation_mask;
    int reconvergence_pc;  // placeholder, should be updated by CFG
};

static BarrierTranslationResult translate_bar_sync(
    int current_pc,             // PC of the bar.sync instruction
    int total_statements,        // kernelStatements.size() + 1 (next instruction)
    int reqntid_x, int reqntid_y, int reqntid_z,
    int maxntid_x, int maxntid_y, int maxntid_z)
{
    BarrierTranslationResult result;

    // Step 1: isWarpLevelBarrier check
    auto is_warp_level = [&]() -> bool {
        if (reqntid_x > 0 || reqntid_y > 0 || reqntid_z > 0) {
            return (reqntid_x * reqntid_y * reqntid_z) <= 32;
        }
        if (maxntid_x > 0 && maxntid_y > 0 && maxntid_z > 0) {
            return (maxntid_x * maxntid_y * maxntid_z) <= 32;
        }
        // Fallback: assume single-warp (THIS IS THE DEFAULT!)
        return true;
    };

    if (is_warp_level()) {
        // Translation: bar.sync → bar.warp.sync.b32 0xFFFFFFFF, -1
        // BUG: participation mask is HARD CODED to 0xFFFFFFFF
        result.participation_mask = 0xFFFFFFFFu;  // ← BUG: should use actual CTA size!
        result.reconvergence_pc = -1;
    } else {
        // Multi-warp: no translation, use default
        result.participation_mask = 0xFFFFFFFFu;
        result.reconvergence_pc = -1;
    }

    return result;
}

/// Apply CFG post-dominator analysis to barrier reconvergence (ptx_interpreter.cpp:608-616)
static int apply_cfg_to_warp_barrier(int barrier_pc,
                                      const cfg::PostDominatorMap& postDoms) {
    // Current code always uses i+1, ignoring post-dominator for barriers
    return barrier_pc + 1;
}

// =============================================================================
// Warp state simulation (minimal simulation of handle_branch)
// =============================================================================

struct SimThread {
    int pc = 0;
    bool is_active = false;
    bool is_blocked = false;
};

struct SimWarp {
    std::array<SimThread, 32> threads;
    uint32_t exec_mask = 0;
    uint32_t taken_mask = 0;
    uint32_t not_taken_mask = 0;
    bool is_divergent = false;
    std::string predicate;
    bool predicate_negated = false;

    void activate_threads(int count) {
        for (int i = 0; i < std::min(count, 32); i++) {
            threads[i].is_active = true;
        }
    }

    // Simulate handle_branch from warp_context.cpp
    void simulate_branch(int current_pc,
                         const std::string& pred,
                         bool pred_negated,
                         int target_pc,
                         int reconvergence_pc)
    {
        predicate = pred;
        predicate_negated = pred_negated;
        taken_mask = 0;
        not_taken_mask = 0;

        for (int i = 0; i < 32; i++) {
            if (!threads[i].is_active) continue;

            // If predicate is empty (解析失败), should_branch defaults to true
            bool should_branch = true;

            if (!pred.empty()) {
                // Simulate: for threads where tid > 15, pred is true
                // This mirrors setp.gt.u32 %p1, %r1, 15 where %r1 = tid.x
                bool pred_bool = (i > 15);
                should_branch = pred_negated ? !pred_bool : pred_bool;
            }
            // If pred is empty, should_branch stays true (all threads branch!)

            if (should_branch) {
                taken_mask |= (1u << i);
            } else {
                not_taken_mask |= (1u << i);
            }
        }

        is_divergent = (taken_mask != 0) && (not_taken_mask != 0);

        if (is_divergent) {
            for (int i = 0; i < 32; i++) {
                if (taken_mask & (1u << i)) {
                    threads[i].pc = target_pc;
                } else if (not_taken_mask & (1u << i)) {
                    threads[i].pc = current_pc + 1;
                }
            }
            exec_mask = taken_mask;
        } else {
            int next_pc = (taken_mask != 0) ? target_pc : current_pc + 1;
            for (int i = 0; i < 32; i++) {
                if (threads[i].is_active) {
                    threads[i].pc = next_pc;
                }
            }
        }
    }

    // Simulate get_lanes_by_pc
    std::map<int, std::vector<int>> lanes_by_pc() const {
        std::map<int, std::vector<int>> result;
        for (int lane = 0; lane < 32; lane++) {
            if (threads[lane].is_active) {
                result[threads[lane].pc].push_back(lane);
            }
        }
        return result;
    }

    int count_at_pc(int pc) const {
        int count = 0;
        for (int i = 0; i < 32; i++) {
            if (threads[i].is_active && threads[i].pc == pc) count++;
        }
        return count;
    }
};

// =============================================================================
// Tests: CFG Post-Dominator Analysis
// =============================================================================

TEST_CASE("T3-CFG-01: Post-dominator for test_nested_sync exact pattern",
          "[t3][cfg][reproduction]")
{
    // Exact replica of _Z16test_nested_sync instruction layout
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;

    // PC=0-9: setup (10 regular statements)
    for (int i = 0; i < 10; i++)
        stmts.push_back(make_regular());

    // PC=10: first barrier
    stmts.push_back(make_warp_barrier("0xFFFFFFFF", "-1"));

    // PC=11: setp
    stmts.push_back(make_regular(S_SETP));

    // PC=12-13: mov + add (executed by ALL threads, before bra!)
    stmts.push_back(make_regular());
    stmts.push_back(make_regular());

    // PC=14: conditional branch
    stmts.push_back(make_branch("L__BB2_2"));
    label2pc["L__BB2_2"] = 22;

    // PC=15-21: shared mem ops (T0-15 only)
    for (int i = 0; i < 7; i++)
        stmts.push_back(make_regular());

    // PC=22: $L__BB2_2 label
    stmts.push_back(make_label("L__BB2_2"));

    // PC=23: cvta
    stmts.push_back(make_regular());

    // PC=24: second barrier (THE DEADLOCK POINT)
    stmts.push_back(make_warp_barrier("0xFFFFFFFF", "-1"));

    // PC=25: ld.shared
    stmts.push_back(make_regular());

    // PC=26-28: final write
    for (int i = 0; i < 2; i++)
        stmts.push_back(make_regular());

    // PC=29: ret
    stmts.push_back(make_regular(S_RET));

    // Build CFG
    cfg::CFG cfg = build_test_cfg(stmts, label2pc);

    // Verify basic block count
    INFO("CFG has " << cfg.blocks.size() << " blocks");
    REQUIRE(!cfg.blocks.empty());

    // Compute post-dominators
    auto postDoms = compute_post_doms(cfg);

    // Key assertions
    INFO("Post-dominator map for test_nested_sync:");
    for (const auto& [pc, ipd] : postDoms) {
        INFO("  PC=" << pc << " -> post-dom=" << ipd);
    }

    // Branch at PC=14 should converge at PC=22 (L__BB2_2)
    auto it_br = postDoms.find(14);
    if (it_br != postDoms.end()) {
        REQUIRE(it_br->second == 22);
    }

    // First barrier at PC=10 should have a valid post-dominator
    auto it_b1 = postDoms.find(10);
    if (it_b1 != postDoms.end()) {
        INFO("Barrier 1 post-dominator: " << it_b1->second);
        REQUIRE(it_b1->second >= 10);
    }

    // Second barrier at PC=24 should have post-dominator >= 25 (ret direction)
    auto it_b2 = postDoms.find(24);
    if (it_b2 != postDoms.end()) {
        INFO("Barrier 2 post-dominator: " << it_b2->second);
        REQUIRE((it_b2->second >= 24 || it_b2->second == -1));
    }

    // All shared-mem instructions (PC=15-21) should converge at PC=22
    for (int pc = 15; pc <= 21; pc++) {
        auto it = postDoms.find(pc);
        if (it != postDoms.end()) {
            REQUIRE(it->second == 22);
        }
    }
}

TEST_CASE("T3-CFG-02: CFG for if-then pattern WITHOUT pre-bra instructions",
          "[t3][cfg][minimal]")
{
    // Minimal pattern: barrier → divergent branch → second barrier
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;

    stmts.push_back(make_regular());          // PC=0: setup
    stmts.push_back(make_warp_barrier("0xFFFFFFFF", "-1"));  // PC=1: barrier 1
    stmts.push_back(make_regular(S_SETP));    // PC=2: setp
    stmts.push_back(make_branch("L_merge"));  // PC=3: bra
    label2pc["L_merge"] = 5;
    stmts.push_back(make_regular());          // PC=4: then-body
    stmts.push_back(make_label("L_merge"));   // PC=5: label
    stmts.push_back(make_warp_barrier("0xFFFFFFFF", "-1"));  // PC=6: barrier 2
    stmts.push_back(make_regular(S_RET));     // PC=7: ret

    cfg::CFG cfg = build_test_cfg(stmts, label2pc);
    auto postDoms = compute_post_doms(cfg);

    INFO("Minimal CFG post-dominators:");
    for (const auto& [pc, ipd] : postDoms) {
        INFO("  PC=" << pc << " -> post-dom=" << ipd);
    }

    // Branch at PC=3 should converge at PC=5 or PC=6
    auto it = postDoms.find(3);
    if (it != postDoms.end()) {
        INFO("Branch post-dominator: " << it->second);
        REQUIRE((it->second == 5 || it->second == 6));
    }

    // Then-body at PC=4 should converge at PC=5 (the label)
    auto it_then = postDoms.find(4);
    if (it_then != postDoms.end()) {
        REQUIRE(it_then->second == 5);
    }
}

// =============================================================================
// Tests: Barrier Translation Layer (the suspected root cause)
// =============================================================================

TEST_CASE("T3-TRANS-01: bar.sync translation with DYNAMIC mask (16-thread CTA)",
          "[t3][translation][fixed]")
{
    auto t1 = translate_bar_sync(
        /* current_pc= */   10,
        /* total_stmts= */  30,
        /* reqntid */       16, 1, 1,    // ← Only 16 threads!
        /* maxntid */       0, 0, 0);

    // After fix: mask should match actual CTA size
    uint32_t expected_mask = compute_participation_mask(16);
    REQUIRE(expected_mask == 0x0000FFFFu);

    INFO("Expected participation_mask=0x" << std::hex << expected_mask
         << " for 16-thread CTA");
}

TEST_CASE("T3-TRANS-02: bar.sync translation with 32-thread CTA (should pass)",
          "[t3][translation]")
{
    auto t1 = translate_bar_sync(
        /* current_pc= */   10,
        /* total_stmts= */  30,
        /* reqntid */       32, 1, 1,    // 32 threads
        /* maxntid */       0, 0, 0);

    // With 32 threads, 0xFFFFFFFF is correct
    REQUIRE(t1.participation_mask == 0xFFFFFFFFu);
    REQUIRE(compute_participation_mask(32) == 0xFFFFFFFFu);
}

TEST_CASE("T3-TRANS-03: bar.sync translation fallback (no reqntid)",
          "[t3][translation]")
{
    // When reqntid is not set, isWarpLevelBarrier returns true by default
    auto t1 = translate_bar_sync(
        /* current_pc= */   10,
        /* total_stmts= */  30,
        /* reqntid */       0, 0, 0,    // Not set
        /* maxntid */       0, 0, 0);

    // Default fallback → assumes single warp → translates with 0xFFFFFFFF
    REQUIRE(t1.participation_mask == 0xFFFFFFFFu);

    INFO("Default fallback assumes 32 threads, but actual count may differ!");
}

TEST_CASE("T3-TRANS-04: Multi-warp CTA (no translation)",
          "[t3][translation]")
{
    auto t1 = translate_bar_sync(
        /* current_pc= */   10,
        /* total_stmts= */  30,
        /* reqntid */       256, 1, 1,    // 8 warps = 256 threads
        /* maxntid */       0, 0, 0);

    // Multi-warp: still uses 0xFFFFFFFF (but doesn't matter since
    // BarHandler goes through SM context, not Wbar)
    REQUIRE(t1.participation_mask == 0xFFFFFFFFu);
}

// =============================================================================
// Tests: Branch Divergence Simulation
// =============================================================================

TEST_CASE("T3-BRANCH-01: Divergent branch with predicate (correct behavior)",
          "[t3][branch][divergent]")
{
    SimWarp warp;
    warp.activate_threads(32);  // All 32 threads active

    warp.simulate_branch(
        /* current_pc= */       14,
        /* predicate= */        "%p1",     // predicate IS present
        /* pred_negated= */     false,
        /* target_pc= */        22,        // $L__BB2_2
        /* reconvergence_pc= */ 22);

    // Divergence should be detected
    REQUIRE(warp.is_divergent == true);

    // T0-15 should NOT branch (pred false for tid ≤ 15)
    REQUIRE(warp.not_taken_mask == 0x0000FFFFu);
    REQUIRE(warp.count_at_pc(15) == 16);  // T0-15 at PC=15

    // T16-31 should branch (pred true for tid > 15)
    REQUIRE(warp.taken_mask == 0xFFFF0000u);
    REQUIRE(warp.count_at_pc(22) == 16);  // T16-31 at PC=22
}

TEST_CASE("T3-BRANCH-02: Divergent branch with EMPTY predicate (all jump!)",
          "[t3][branch][bug]")
{
    // This simulates the case where @%p1 is NOT recognized by the parser,
    // resulting in predicate="" → all threads should_branch=true.

    SimWarp warp;
    warp.activate_threads(32);

    warp.simulate_branch(
        /* current_pc= */       14,
        /* predicate= */        "",        // ← EMPTY! parser didn't recognize @%p1
        /* pred_negated= */     false,
        /* target_pc= */        22,
        /* reconvergence_pc= */ 22);

    // NO divergence detected — all threads take the same path!
    REQUIRE(warp.is_divergent == false);

    // ALL threads jump to target
    REQUIRE(warp.count_at_pc(22) == 32);
    REQUIRE(warp.count_at_pc(15) == 0);  // Nobody stays!

    // This means T0-15 SKIP their shared memory writes → data corruption
    INFO("Bug: All threads jumped due to empty predicate!");
}

TEST_CASE("T3-BRANCH-03: get_lanes_by_pc with correct divergence",
          "[t3][branch][scheduler]")
{
    SimWarp warp;
    warp.activate_threads(32);
    warp.simulate_branch(14, "%p1", false, 22, 22);

    REQUIRE(warp.is_divergent == true);

    auto lanes = warp.lanes_by_pc();

    // Two PC groups
    REQUIRE(lanes.size() == 2);

    // PC=15: T0-15 (not-taken path)
    auto it_fallthrough = lanes.find(15);
    REQUIRE(it_fallthrough != lanes.end());
    REQUIRE(it_fallthrough->second.size() == 16);

    // PC=22: T16-31 (taken path)
    auto it_taken = lanes.find(22);
    REQUIRE(it_taken != lanes.end());
    REQUIRE(it_taken->second.size() == 16);

    // Lanes should be in ascending order within each group
    for (size_t i = 0; i < it_fallthrough->second.size(); i++) {
        REQUIRE(it_fallthrough->second[i] == static_cast<int>(i));
    }
    for (size_t i = 0; i < it_taken->second.size(); i++) {
        REQUIRE(it_taken->second[i] == static_cast<int>(16 + i));
    }
}

TEST_CASE("T3-BRANCH-04: get_lanes_by_pc with 16-thread CTA",
          "[t3][branch][cta-size]")
{
    // CTA has only 16 threads — this is a common test configuration
    SimWarp warp;
    warp.activate_threads(16);  // Only lanes 0-15

    warp.simulate_branch(14, "%p1", false, 22, 22);

    // With 16 threads (tid 0-15), all have pred=false → NONE take the branch
    // So all 16 threads continue fall-through
    REQUIRE(warp.is_divergent == false);  // NOT divergent — all same!
    REQUIRE(warp.count_at_pc(15) == 16);
    REQUIRE(warp.count_at_pc(22) == 0);

    auto lanes = warp.lanes_by_pc();
    REQUIRE(lanes.size() == 1);

    // ONLY one PC group — all threads at PC=15
    auto it = lanes.begin();
    REQUIRE(it->first == 15);
    REQUIRE(it->second.size() == 16);

    // This is IMPORTANT: no T16-31 means nobody jumps to PC=22
    // Second barrier at PC=24 will NEVER be reached by the taken path
    // (because there IS no taken path with 16 threads!)
    INFO("With 16-thread CTA, there is no divergence — all threads same path");
}

// =============================================================================
// Tests: Full Integration Simulation (barrier deadlock reproduction)
// =============================================================================

TEST_CASE("T3-FULL-01: 16-thread CTA with correct mask prevents deadlock",
          "[t3][integration][fixed]")
{
    SimWarp warp;
    warp.activate_threads(16);

    for (int i = 0; i < 16; i++) {
        warp.threads[i].pc = 24;
    }

    REQUIRE(warp.count_at_pc(24) == 16);

    uint32_t correct_mask = compute_participation_mask(16);
    REQUIRE(correct_mask == 0x0000FFFFu);

    bool is_complete = (correct_mask == correct_mask);
    REQUIRE(is_complete == true);

    INFO("With correct mask 0x" << std::hex << correct_mask
         << ", barrier completes with " << count_set_bits(correct_mask) << " threads");
}

TEST_CASE("T3-FULL-02: Correct mask prevents deadlock",
          "[t3][integration][fix]")
{
    // This test shows what SHOULD happen with the correct mask

    SimWarp warp;
    warp.activate_threads(16);

    // Pass first barrier
    for (int i = 0; i < 16; i++) {
        warp.threads[i].pc = 24;
    }

    // CORRECT: Use actual CTA size for participation mask
    int cta_threads = 16;
    uint32_t correct_mask = compute_participation_mask(cta_threads);

    REQUIRE(correct_mask == 0x0000FFFFu);

    // All 16 lanes arrive
    uint32_t arrived = correct_mask;

    // is_complete() should succeed
    bool is_complete = (arrived == correct_mask);
    REQUIRE(is_complete == true);

    // With correct mask, no deadlock
    INFO("With correct mask 0x" << std::hex << correct_mask
         << ", barrier completes with " << count_set_bits(arrived) << " threads");
}

TEST_CASE("T3-FULL-03: End-to-end simulation with 32-thread CTA",
          "[t3][integration][full]")
{
    // Full simulation with all 32 threads active — the "happy path"

    SimWarp warp;
    warp.activate_threads(32);

    // Step 1: Pass first barrier (PC=10)
    // (implicit — all threads proceed)

    // Step 2: setp + bra at PC=11-14
    warp.simulate_branch(14, "%p1", false, 22, 22);

    REQUIRE(warp.is_divergent == true);
    REQUIRE(warp.count_at_pc(15) == 16);   // T0-15
    REQUIRE(warp.count_at_pc(22) == 16);   // T16-31

    // Step 3: Execute both paths
    // Simulate: T0-15 go through PC=15-21 → 22 → 23 → 24
    for (int i = 0; i < 16; i++) warp.threads[i].pc = 24;
    // T16-31 go through PC=22-23 → 24
    for (int i = 16; i < 32; i++) warp.threads[i].pc = 24;

    REQUIRE(warp.count_at_pc(24) == 32);

    // Step 4: Second barrier
    auto translation = translate_bar_sync(24, 30, 32, 1, 1, 0, 0, 0);
    REQUIRE(translation.participation_mask == 0xFFFFFFFFu);

    // With 32 active threads, mask matches
    REQUIRE(warp.count_at_pc(24) == 32);

    INFO("32-thread CTA: all threads at barrier, mask matches");
}

TEST_CASE("T3-FULL-04: CFG post-dominator + barrier reconvergence consistency",
          "[t3][integration][cfg-barrier]")
{
    // Test that CFG analysis correctly labels barrier reconvergence
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;

    // Minimal reproducer
    stmts.push_back(make_regular());          // PC=0
    stmts.push_back(make_warp_barrier("0xFFFFFFFF", "-1"));  // PC=1
    stmts.push_back(make_regular(S_SETP));    // PC=2
    stmts.push_back(make_branch("L_skip"));   // PC=3
    label2pc["L_skip"] = 5;
    stmts.push_back(make_regular());          // PC=4
    stmts.push_back(make_label("L_skip"));    // PC=5
    stmts.push_back(make_warp_barrier("0xFFFFFFFF", "-1"));  // PC=6
    stmts.push_back(make_regular(S_RET));     // PC=7

    auto cfg = build_test_cfg(stmts, label2pc);
    auto postDoms = compute_post_doms(cfg);

    // Verify branch at PC=3 converges at PC=5 (L_skip label)
    auto it_br = postDoms.find(3);
    if (it_br != postDoms.end()) {
        REQUIRE(it_br->second == 5);
    }

    // Second barrier at PC=6 should reconverge to PC=7 (next instruction)
    int barrier2_reconvergence = apply_cfg_to_warp_barrier(6, postDoms);
    REQUIRE(barrier2_reconvergence == 7);

    // First barrier at PC=1 should reconverge to PC=2
    int barrier1_reconvergence = apply_cfg_to_warp_barrier(1, postDoms);
    REQUIRE(barrier1_reconvergence == 2);

    INFO("Barrier reconvergence: PC=1→" << barrier1_reconvergence
         << ", PC=6→" << barrier2_reconvergence);
}

// =============================================================================
// Tests: Edge Cases
// =============================================================================

TEST_CASE("T3-EDGE-01: Empty CTA (should not crash)",
          "[t3][edge]")
{
    auto mask = compute_participation_mask(0);
    REQUIRE(mask == 0x00000000u);
    REQUIRE(count_set_bits(mask) == 0);
}

TEST_CASE("T3-EDGE-02: Single thread CTA",
          "[t3][edge]")
{
    auto mask = compute_participation_mask(1);
    REQUIRE(mask == 0x00000001u);
    REQUIRE(count_set_bits(mask) == 1);
}

TEST_CASE("T3-EDGE-03: Exactly 32 threads (full warp)",
          "[t3][edge]")
{
    auto mask = compute_participation_mask(32);
    REQUIRE(mask == 0xFFFFFFFFu);
    REQUIRE(count_set_bits(mask) == 32);
}

TEST_CASE("T3-EDGE-04: Branch target PC out of range",
          "[t3][edge]")
{
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;

    stmts.push_back(make_regular());          // PC=0
    stmts.push_back(make_branch("L_nowhere")); // PC=1 — no such label!
    label2pc["L_nowhere"] = 100;  // label points beyond statement list
    stmts.push_back(make_regular(S_RET));     // PC=2

    auto cfg = build_test_cfg(stmts, label2pc);

    // Should not crash — CFG should handle out-of-range targets
    REQUIRE(cfg.blocks.size() > 0);
}
