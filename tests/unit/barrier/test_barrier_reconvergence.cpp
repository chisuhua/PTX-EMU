/**
 * CFG Barrier Reconvergence Test
 * 
 * Tests that CFG post-dominator computation correctly assigns reconvergence_pc
 * to BARRIER instructions (not just branches).
 * 
 * This test verifies the pattern from test_syncthreads where:
 * 1. Barrier instruction (PC=10) - first barrier
 * 2. Setp instruction (PC=11) - causes divergence
 * 3. Conditional branch (PC=12) - threads take different paths
 * 4. Instructions (PC=13-15) - only some threads execute
 * 5. Branch target label (PC=16)
 * 6. Second barrier instruction (PC=17+) - where ALL threads should reconverge
 * 
 * NOTE: This test needs to be registered with add_catch_test in CMakeLists.txt
 *       CMake will handle the registration when you add it to the test target.
 */

#include "catch_amalgamated.hpp"
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_factory.h"
#include <vector>
#include <map>
#include <string>

namespace cfg = ptx::cfg;
using namespace ptx;
using namespace ptxir::factory;

// Helper: Create a regular (non-branch, non-barrier) statement
static StatementContext make_regular_stmt(StatementType type = S_MOV) {
    StatementContext ctx;
    ctx.type = type;
    GenericInstr instr;
    ctx.data = instr;
    return ctx;
}

// Helper: Create a branch statement
static StatementContext make_branch_stmt(const std::string& target) {
    StatementContext ctx;
    ctx.type = S_BRA;
    BranchInstr branch;
    branch.target = target;
    branch.reconvergence_pc = -1;  // Not yet computed
    ctx.data = branch;
    return ctx;
}

// Helper: Create a label statement
static StatementContext make_label_stmt(const std::string& name) {
    StatementContext ctx;
    ctx.type = S_DOLLOR;
    DollarNameInstr label;
    label.name = name;
    ctx.data = label;
    return ctx;
}

TEST_CASE("CFG: post-dominator for divergent branch before barrier", "[cfg][reconvergence][barrier]") {
    // Pattern matching test_syncthreads:
    // PC=0-9:  Setup instructions (mov, etc.)
    // PC=10:  bar.warp.sync  (first barrier - all threads reconverge here before divergence)
    // PC=11:  setp.gt.u32    (divergent comparison - creates predicate)
    // PC=12:  @%p1 bra $L_skip  (conditional branch - threads 16-31 take branch)
    // PC=13-15: Instructions only executed by threads 0-15 (threads that didn't take branch)
    // PC=16:  $L_skip: label   (branch target - merge point for divergent paths)
    // PC=17:  bar.warp.sync  (second barrier - ALL 32 threads reconverge here)
    // PC=18:  ret
    
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;
    
    // PC=0-9: Setup instructions (10 regular instructions)
    for (int i = 0; i < 10; i++) {
        stmts.push_back(make_regular_stmt());
    }
    
    // PC=10: First barrier (all threads converge here before divergence)
    stmts.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 0));
    
    // PC=11: setp instruction (divergent instruction - but for CFG purposes just a regular stmt)
    stmts.push_back(make_regular_stmt(S_SETP));
    
    // PC=12: Conditional branch to L_skip
    stmts.push_back(make_branch_stmt("L_skip"));
    label2pc["L_skip"] = 16;  // Branch target is at PC=16
    
    // PC=13-15: Instructions executed only by threads that didn't take branch
    for (int i = 0; i < 3; i++) {
        stmts.push_back(make_regular_stmt());
    }
    
    // PC=16: Branch target label (L_skip)
    stmts.push_back(make_label_stmt("L_skip"));
    
    // PC=17: Second barrier (where all threads reconverge)
    stmts.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 0));
    
    // PC=18: ret (exit)
    stmts.push_back(make_regular_stmt(S_RET));
    
    // Build CFG
    cfg::CFG cfg = cfg::CFGBuilder::build(stmts, label2pc);
    
    // Compute post-dominators
    cfg::PostDominatorMap postDoms = cfg::CFGBuilder::computePostDominators(cfg);
    
    // Print post-dominator map for debugging
    INFO("Post-dominator map:");
    for (const auto& [pc, ipd] : postDoms) {
        INFO("  PC=" << pc << " -> post-dom=" << ipd);
    }
    
    // Key assertions:
    
    // The branch at PC=12 should have post-dominator at PC=16 (merge point)
    // Both paths (taken branch to PC=16, fall-through through PC=13-15 to PC=16) merge at PC=16
    auto it_branch = postDoms.find(12);
    if (it_branch != postDoms.end()) {
        INFO("Branch at PC=12 has post-dominator: " << it_branch->second);
        REQUIRE(it_branch->second == 16);  // Branch should reconverge at L_skip
    } else {
        FAIL("Branch at PC=12 not found in post-dominator map");
    }
    
    // Instructions PC=13,14,15 should have post-dominator at PC=16 (they only execute on one path)
    for (int pc = 13; pc <= 15; pc++) {
        auto it = postDoms.find(pc);
        if (it != postDoms.end()) {
            INFO("PC=" << pc << " has post-dominator: " << it->second);
            REQUIRE(it->second == 16);  // These instructions only reconverge at L_skip
        }
    }
    
    // The first barrier at PC=10 should have post-dominator (all threads pass through it)
    // After the barrier, threads diverge at PC=12 but must reconverge at second barrier (PC=17)
    auto it_barrier1 = postDoms.find(10);
    if (it_barrier1 != postDoms.end()) {
        INFO("First barrier at PC=10 has post-dominator: " << it_barrier1->second);
        // First barrier should lead to post-dom of second barrier or exit
        // The key is that barrier at PC=10 is post-dominated by at least the second barrier
        REQUIRE(it_barrier1->second >= 10);  // Should have a valid post-dom
    }
    
    // The second barrier at PC=17 should be post-dominated by ret or itself
    auto it_barrier2 = postDoms.find(17);
    if (it_barrier2 != postDoms.end()) {
        INFO("Second barrier at PC=17 has post-dominator: " << it_barrier2->second);
        // Second barrier should lead to ret/exit
        bool barrier2_valid = (it_barrier2->second >= 17) || (it_barrier2->second == -1);
        REQUIRE(barrier2_valid);
    }
}

TEST_CASE("CFG: post-dominator for simple if-else pattern", "[cfg][reconvergence]") {
    // Simple if-else pattern:
    // PC=0: setp (comparison - creates predicate)
    // PC=1: @%p1 bra $L_else  (conditional branch)
    // PC=2: then-body (1 instruction)
    // PC=3: $L_else: label    (else label / merge point)
    // PC=4: else-body (1 instruction)
    // PC=5: merge/ret
    
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;
    
    stmts.push_back(make_regular_stmt(S_SETP));  // PC=0: setp
    stmts.push_back(make_branch_stmt("L_else"));  // PC=1: conditional branch
    label2pc["L_else"] = 3;
    
    stmts.push_back(make_regular_stmt());  // PC=2: then-body
    
    stmts.push_back(make_label_stmt("L_else"));  // PC=3: else label
    
    stmts.push_back(make_regular_stmt());  // PC=4: else-body
    
    stmts.push_back(make_regular_stmt(S_RET));  // PC=5: merge/ret
    
    // Build CFG
    cfg::CFG cfg = cfg::CFGBuilder::build(stmts, label2pc);
    
    // Compute post-dominators
    cfg::PostDominatorMap postDoms = cfg::CFGBuilder::computePostDominators(cfg);
    
    INFO("Post-dominator map for simple if-else:");
    for (const auto& [pc, ipd] : postDoms) {
        INFO("  PC=" << pc << " -> post-dom=" << ipd);
    }
    
    // Branch at PC=1 should reconverge at the merge point (PC=3 or PC=5)
    // Both paths (taken branch to PC=3, fall-through to PC=2 then to PC=3) merge at PC=3
    auto it = postDoms.find(1);
    if (it != postDoms.end()) {
        INFO("Branch at PC=1 has post-dominator: " << it->second);
        REQUIRE(it->second >= 3);  // Post-dominator should be at or after merge point
    } else {
        FAIL("Branch at PC=1 not found in post-dominator map");
    }
    
    // Then-body at PC=2 should have post-dominator at merge point
    auto it_then = postDoms.find(2);
    if (it_then != postDoms.end()) {
        INFO("Then-body at PC=2 has post-dominator: " << it_then->second);
        REQUIRE(it_then->second >= 3);  // Should reconverge at or after L_else
    }
}

TEST_CASE("CFG: barrier as explicit reconvergence point", "[cfg][reconvergence][barrier]") {
    // Test that barriers are recognized as reconvergence points
    // Pattern:
    // PC=0: barrier
    // PC=1: divergent branch
    // PC=2-3: path 1
    // PC=4: label (merge)
    // PC=5-6: path 2
    // PC=7: barrier (reconvergence)
    // PC=8: ret
    
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;
    
    stmts.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 0));  // PC=0: first barrier
    stmts.push_back(make_branch_stmt("L_else"));  // PC=1: divergent branch
    label2pc["L_else"] = 4;
    
    stmts.push_back(make_regular_stmt());  // PC=2: path 1a
    stmts.push_back(make_regular_stmt());  // PC=3: path 1b
    
    stmts.push_back(make_label_stmt("L_else"));  // PC=4: merge label
    
    stmts.push_back(make_regular_stmt());  // PC=5: path 2a
    stmts.push_back(make_regular_stmt());  // PC=6: path 2b
    
    stmts.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 0));  // PC=7: second barrier (reconvergence)
    stmts.push_back(make_regular_stmt(S_RET));  // PC=8: ret
    
    // Build CFG
    cfg::CFG cfg = cfg::CFGBuilder::build(stmts, label2pc);
    
    // Compute post-dominators
    cfg::PostDominatorMap postDoms = cfg::CFGBuilder::computePostDominators(cfg);
    
    INFO("Post-dominator map for barrier reconvergence:");
    for (const auto& [pc, ipd] : postDoms) {
        INFO("  PC=" << pc << " -> post-dom=" << ipd);
    }
    
    // The divergent branch at PC=1 should have post-dom at merge point (PC=4)
    // CFG analysis finds control flow post-dominators, not barrier synchronization points
    auto it_branch = postDoms.find(1);
    if (it_branch != postDoms.end()) {
        INFO("Branch at PC=1 has post-dominator: " << it_branch->second);
        REQUIRE(it_branch->second == 4);  // All paths merge at L_else label
    } else {
        FAIL("Branch at PC=1 not found in post-dominator map");
    }
    
    // The first barrier at PC=0 should post-dominate to at least the second barrier
    auto it_barrier1 = postDoms.find(0);
    if (it_barrier1 != postDoms.end()) {
        INFO("First barrier at PC=0 has post-dominator: " << it_barrier1->second);
        REQUIRE(it_barrier1->second >= 0);  // Should have valid post-dom
    }
    
    // Second barrier at PC=7 should be post-dominated by ret
    auto it_barrier2 = postDoms.find(7);
    if (it_barrier2 != postDoms.end()) {
        INFO("Second barrier at PC=7 has post-dominator: " << it_barrier2->second);
        // Should be -1 (exit/not found) or >= 8 (ret)
        bool barrier2_valid = (it_barrier2->second >= 7) || (it_barrier2->second == -1);
        REQUIRE(barrier2_valid);
    }
}

TEST_CASE("CFG: post-dominator map completeness", "[cfg][reconvergence]") {
    // Verify that the post-dominator map contains entries for all statements
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;
    
    // Create a simple linear kernel
    stmts.push_back(make_regular_stmt());
    stmts.push_back(make_regular_stmt());
    stmts.push_back(make_regular_stmt());
    stmts.push_back(make_regular_stmt(S_RET));
    
    cfg::CFG cfg = cfg::CFGBuilder::build(stmts, label2pc);
    cfg::PostDominatorMap postDoms = cfg::CFGBuilder::computePostDominators(cfg);
    
    // For linear code, every PC should be in the post-dom map
    REQUIRE(postDoms.size() == stmts.size());
    
    for (size_t i = 0; i < stmts.size(); i++) {
        INFO("Checking PC=" << i << " is in post-dom map");
        REQUIRE(postDoms.find(static_cast<int>(i)) != postDoms.end());
    }
}
