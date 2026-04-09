#pragma once

#include <vector>
#include <map>
#include <set>
#include <string>

#include "ptx_ir/statement_context.h"

namespace ptx {
namespace cfg {

struct BasicBlock {
    int id;
    int start_pc;
    int end_pc;
    
    std::vector<int> successors;
    std::vector<int> predecessors;
    
    bool is_branch_target;
    bool is_exit;
    
    int size() const { return end_pc - start_pc; }
    bool contains(int pc) const { return pc >= start_pc && pc < end_pc; }
    bool empty() const { return size() == 0; }
};

struct CFG {
    std::vector<BasicBlock> blocks;
    int entry_block_id;
    int exit_block_id;
    
    BasicBlock* find_block_by_pc(int pc);
    BasicBlock* find_block_by_id(int id);
    void print() const;
};

using PostDominatorMap = std::map<int, int>;

class CFGBuilder {
public:
    static CFG build(
        const std::vector<StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    static PostDominatorMap computePostDominators(const CFG& cfg);
    
private:
    static std::vector<BasicBlock> identifyBasicBlocks(
        const std::vector<StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    static std::set<int> findBranchTargets(
        const std::vector<StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    static void buildEdges(CFG& cfg,
                          const std::vector<StatementContext>& statements);
    
    static int findImmediatePostDominator(
        const BasicBlock& block,
        const std::map<int, std::set<int>>& postDomSets);
};

}
}
