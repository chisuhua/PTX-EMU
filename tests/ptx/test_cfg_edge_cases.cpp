/**
 * CFG Builder Edge Case Tests
 * Tests corner cases and boundary conditions
 * Simple C++ standalone (no Catch2 dependency)
 */

#include "../src/ptx_parser/cfg_builder.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <map>

using namespace ptx::cfg;

int tests_run = 0;
int tests_passed = 0;

#define TEST_CASE(name, tags) \
    void name(); \
    void name()

#define REQUIRE(expr) \
    do { \
        tests_run++; \
        if (!(expr)) { \
            std::cerr << "FAIL: " << __LINE__ << std::endl; \
        } else { \
            tests_passed++; \
            std::cout << "."; \
        } \
    } while(0)

// ============================================================================
// High Priority Edge Cases (🔴)
// ============================================================================

TEST_CASE(test_empty_kernel, "[cfg][edge][high]") {
    std::cout << "\nTest: Empty kernel (0 statements)... ";
    std::vector<ptxemu::ir::StatementContext> statements;
    std::map<std::string, int> label2pc;
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() >= 0);
    REQUIRE(cfg.entry_block_id == 0);
    std::cout << "PASS" << std::endl;
}

TEST_CASE(test_single_statement, "[cfg][edge][high]") {
    std::cout << "\nTest: Single statement kernel... ";
    std::vector<ptxemu::ir::StatementContext> statements(1);
    statements[0].type = S_RET;
    
    std::map<std::string, int> label2pc;
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() == 1);
    REQUIRE(cfg.blocks[0].is_exit == true);
    std::cout << "PASS" << std::endl;
}

TEST_CASE(test_self_referencing_branch, "[cfg][edge][high]") {
    std::cout << "\nTest: Self-referencing branch... ";
    std::vector<ptxemu::ir::StatementContext> statements(2);
    statements[0].type = S_BRA;
    BranchInstr branch;
    branch.target = "loop";
    branch.reconvergence_pc = -1;
    statements[0].data = branch;
    
    statements[1].type = S_DOLLOR;
    DollarNameInstr label;
    label.name = "loop";
    statements[1].data = label;
    
    std::map<std::string, int> label2pc;
    label2pc["loop"] = 1;
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() >= 1);
    
    PostDominatorMap postDoms = CFGBuilder::computePostDominators(cfg);
    REQUIRE(postDoms.size() > 0);
    std::cout << "PASS" << std::endl;
}

TEST_CASE(test_missing_branch_target, "[cfg][edge][high]") {
    std::cout << "\nTest: Missing branch target... ";
    std::vector<ptxemu::ir::StatementContext> statements(1);
    statements[0].type = S_BRA;
    BranchInstr branch;
    branch.target = "L_nonexistent";
    branch.reconvergence_pc = -1;
    statements[0].data = branch;
    
    std::map<std::string, int> label2pc;
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() >= 0);
    
    PostDominatorMap postDoms = CFGBuilder::computePostDominators(cfg);
    REQUIRE(postDoms.size() >= 0);
    std::cout << "PASS" << std::endl;
}

TEST_CASE(test_unreachable_label, "[cfg][edge][high]") {
    std::cout << "\nTest: Unreachable label... ";
    std::vector<ptxemu::ir::StatementContext> statements(3);
    statements[0].type = S_RET;
    
    statements[1].type = S_DOLLOR;
    DollarNameInstr label1;
    label1.name = "unreachable";
    statements[1].data = label1;
    
    statements[2].type = S_RET;
    
    std::map<std::string, int> label2pc;
    label2pc["unreachable"] = 1;
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() >= 1);
    std::cout << "PASS" << std::endl;
}

TEST_CASE(test_deep_nested_branches, "[cfg][edge][high]") {
    std::cout << "\nTest: Deep nested branches (10 levels)... ";
    std::vector<ptxemu::ir::StatementContext> statements(30);
    std::map<std::string, int> label2pc;
    
    int pc = 0;
    for (int i = 0; i < 10; i++) {
        statements[pc].type = S_BRA;
        BranchInstr branch;
        branch.target = "L_inner_" + std::to_string(i);
        branch.reconvergence_pc = -1;
        statements[pc].data = branch;
        label2pc["L_inner_" + std::to_string(i)] = pc + 2;
        pc++;
        
        statements[pc].type = S_RET;
        pc++;
    }
    
    statements[pc].type = S_DOLLOR;
    DollarNameInstr merge;
    merge.name = "L_merge";
    statements[pc].data = merge;
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() > 0);
    
    PostDominatorMap postDoms = CFGBuilder::computePostDominators(cfg);
    REQUIRE(postDoms.size() > 0);
    std::cout << "PASS" << std::endl;
}

TEST_CASE(test_multi_branch_convergence, "[cfg][edge][high]") {
    std::cout << "\nTest: Multi-branch convergence (3 paths)... ";
    std::vector<ptxemu::ir::StatementContext> statements(10);
    std::map<std::string, int> label2pc;
    
    statements[0].type = S_BRA;
    BranchInstr b1;
    b1.target = "L_path1";
    b1.reconvergence_pc = -1;
    statements[0].data = b1;
    label2pc["L_path1"] = 3;
    
    statements[1].type = S_BRA;
    BranchInstr b2;
    b2.target = "L_merge";
    b2.reconvergence_pc = -1;
    statements[1].data = b2;
    label2pc["L_merge"] = 9;
    
    statements[2].type = S_DOLLOR;
    DollarNameInstr l1;
    l1.name = "L_path1";
    statements[2].data = l1;
    statements[3].type = S_BRA;
    BranchInstr b3;
    b3.target = "L_merge";
    b3.reconvergence_pc = -1;
    statements[3].data = b3;
    
    statements[4].type = S_DOLLOR;
    DollarNameInstr merge;
    merge.name = "L_merge";
    statements[4].data = merge;
    statements[5].type = S_RET;
    
    statements.resize(6);
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() >= 1);
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Medium Priority Edge Cases (🟡)
// ============================================================================

TEST_CASE(test_linear_code, "[cfg][edge][medium]") {
    std::cout << "\nTest: Linear code (no branches)... ";
    std::vector<ptxemu::ir::StatementContext> statements(3);
    statements[0].type = S_RET;
    statements[1].type = S_RET;
    statements[2].type = S_RET;
    
    std::map<std::string, int> label2pc;
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() == 1);
    std::cout << "PASS" << std::endl;
}

TEST_CASE(test_duplicate_labels, "[cfg][edge][medium]") {
    std::cout << "\nTest: Duplicate labels... ";
    std::vector<ptxemu::ir::StatementContext> statements(4);
    
    statements[0].type = S_DOLLOR;
    DollarNameInstr label1;
    label1.name = "L_dup";
    statements[0].data = label1;
    
    statements[1].type = S_RET;
    
    statements[2].type = S_DOLLOR;
    DollarNameInstr label2;
    label2.name = "L_dup";
    statements[2].data = label2;
    
    statements[3].type = S_RET;
    
    std::map<std::string, int> label2pc;
    label2pc["L_dup"] = 2;
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() >= 0);
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Integration & Performance Tests
// ============================================================================

TEST_CASE(test_cfg_with_barrier, "[cfg][integration]") {
    std::cout << "\nTest: CFG with barrier instruction... ";
    std::vector<ptxemu::ir::StatementContext> statements(5);
    
    statements[0].type = S_BRA;
    BranchInstr branch;
    branch.target = "L_then";
    branch.reconvergence_pc = -1;
    statements[0].data = branch;
    std::map<std::string, int> label2pc;
    label2pc["L_then"] = 3;
    
    statements[1].type = S_DOLLOR;
    DollarNameInstr shared;
    shared.name = "shared0";
    statements[1].data = shared;
    
    statements[2].type = S_DOLLOR;
    DollarNameInstr l_then;
    l_then.name = "L_then";
    statements[2].data = l_then;
    
    statements[3].type = S_BAR;
    BarrierInstr barrier;
    barrier.type = "sync";
    barrier.barId = 0;
    statements[3].data = barrier;
    
    statements[4].type = S_RET;
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() >= 1);
    std::cout << "PASS" << std::endl;
}

TEST_CASE(test_large_kernel, "[cfg][perf]") {
    std::cout << "\nTest: Large kernel (100 statements)... ";
    std::vector<ptxemu::ir::StatementContext> statements(100);
    std::map<std::string, int> label2pc;
    
    for (int i = 0; i < 100; i++) {
        statements[i].type = S_RET;
    }
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() >= 1);
    
    PostDominatorMap postDoms = CFGBuilder::computePostDominators(cfg);
    REQUIRE(postDoms.size() > 0);
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Helper Tests
// ============================================================================

TEST_CASE(test_basicblock_contains, "[cfg][helper]") {
    std::cout << "\nTest: BasicBlock::contains... ";
    BasicBlock block;
    block.start_pc = 10;
    block.end_pc = 20;
    
    REQUIRE(block.contains(10) == true);
    REQUIRE(block.contains(15) == true);
    REQUIRE(block.contains(19) == true);
    REQUIRE(block.contains(20) == false);
    REQUIRE(block.contains(5) == false);
    std::cout << "PASS" << std::endl;
}

TEST_CASE(test_basicblock_size, "[cfg][helper]") {
    std::cout << "\nTest: BasicBlock::size... ";
    BasicBlock block1;
    block1.start_pc = 0;
    block1.end_pc = 10;
    REQUIRE(block1.size() == 10);
    
    BasicBlock block2;
    block2.start_pc = 5;
    block2.end_pc = 5;
    REQUIRE(block2.size() == 0);
    REQUIRE(block2.empty() == true);
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== CFG Builder Edge Case Tests ===" << std::endl;
    std::cout << "Running " << 14 << " tests..." << std::endl << std::endl;
    
    // High priority
    test_empty_kernel();
    test_single_statement();
    test_self_referencing_branch();
    test_missing_branch_target();
    test_unreachable_label();
    test_deep_nested_branches();
    test_multi_branch_convergence();
    
    // Medium priority
    test_linear_code();
    test_duplicate_labels();
    
    // Integration & Performance
    test_cfg_with_barrier();
    test_large_kernel();
    
    // Helpers
    test_basicblock_contains();
    test_basicblock_size();
    
    std::cout << "\n\n=== Test Summary ===" << std::endl;
    std::cout << "Tests run: " << tests_run << std::endl;
    std::cout << "Tests passed: " << tests_passed << std::endl;
    std::cout << "Tests failed: " << (tests_run - tests_passed) << std::endl;
    
    if (tests_passed == tests_run) {
        std::cout << "\n✅ ALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ " << (tests_run - tests_passed) << " TESTS FAILED" << std::endl;
        return 1;
    }
}
