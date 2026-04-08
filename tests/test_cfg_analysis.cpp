#include <catch2/catch.hpp>
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/symtable.h"

using namespace ptx::cfg;

TEST_CASE("CFG Builder - Basic Block Identification", "[cfg][basicblock]") {
    std::vector<StatementContext> statements(10);
    std::map<std::string, int> label2pc;
    
    label2pc["target"] = 5;
    
    SECTION("Single block, no branches") {
        auto blocks = CFGBuilder::identifyBasicBlocks(statements, label2pc);
        REQUIRE(blocks.size() == 1);
        REQUIRE(blocks[0].start_pc == 0);
        REQUIRE(blocks[0].end_pc == 10);
    }
    
    SECTION("Multiple blocks with branch") {
        statements[2].type = S_BRA;
        BranchInstr branch;
        branch.target = "target";
        statements[2].data = branch;
        
        auto blocks = CFGBuilder::identifyBasicBlocks(statements, label2pc);
        
        REQUIRE(blocks.size() >= 2);
        REQUIRE(blocks[0].end_pc == 3);
        REQUIRE(blocks[0].is_branch_target == false);
        REQUIRE(blocks[1].start_pc >= 3);
    }
}

TEST_CASE("CFG Builder - Branch Target Detection", "[cfg][targets]") {
    std::vector<StatementContext> statements(10);
    std::map<std::string, int> label2pc;
    
    label2pc["target1"] = 3;
    label2pc["target2"] = 7;
    
    statements[2].type = S_BRA;
    BranchInstr b1;
    b1.target = "target1";
    statements[2].data = b1;
    
    statements[5].type = S_BRA;
    BranchInstr b2;
    b2.target = "target2";
    statements[5].data = b2;
    
    auto targets = CFGBuilder::findBranchTargets(statements, label2pc);
    
    REQUIRE(targets.count(3) == 1);
    REQUIRE(targets.count(7) == 1);
}

TEST_CASE("CFG Builder - Post-Dominator Computation", "[cfg][postdom]") {
    std::vector<StatementContext> statements(10);
    std::map<std::string, int> label2pc;
    
    label2pc["merge"] = 8;
    
    statements[2].type = S_BRA;
    BranchInstr branch;
    branch.target = "merge";
    statements[2].data = branch;
    
    auto postDoms = CFGBuilder::computePostDominators(statements, label2pc);
    
    REQUIRE(postDoms.size() > 0);
    
    for (const auto& [pc, postDom] : postDoms) {
        if (pc < 8) {
            REQUIRE(postDom >= pc);
        }
    }
}

TEST_CASE("CFG Builder - Full CFG Construction", "[cfg][full]") {
    std::vector<StatementContext> statements(20);
    std::map<std::string, int> label2pc;
    
    label2pc["target"] = 5;
    label2pc["end"] = 15;
    
    statements[4].type = S_BRA;
    BranchInstr b1;
    b1.target = "target";
    statements[4].data = b1;
    
    statements[14].type = S_BRA;
    BranchInstr b2;
    b2.target = "end";
    statements[14].data = b2;
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() > 0);
    REQUIRE(cfg.entry_block_id == 0);
    
    bool foundExit = false;
    for (const auto& block : cfg.blocks) {
        if (block.is_exit) {
            foundExit = true;
            break;
        }
    }
    REQUIRE(foundExit);
}
