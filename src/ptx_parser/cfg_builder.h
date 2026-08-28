#pragma once

#include <vector>
#include <map>
#include <set>
#include <string>

#include "ptx_ir/statement_context.h"

namespace ptx {
namespace cfg {

/**
 * @brief Basic Block in Control Flow Graph
 */
struct BasicBlock {
    int id;               ///< Block identifier
    int start_pc;         ///< Starting PC (inclusive)
    int end_pc;           ///< Ending PC (exclusive)
    
    std::vector<int> successors;    ///< IDs of successor blocks
    std::vector<int> predecessors;  ///< IDs of predecessor blocks
    
    bool is_branch_target;  ///< Is this block a branch target?
    bool is_exit;           ///< Is this the exit block?
    
    int size() const { return end_pc - start_pc; }
    bool contains(int pc) const { return pc >= start_pc && pc < end_pc; }
    bool empty() const { return size() == 0; }
};

/**
 * @brief Control Flow Graph representation
 */
struct CFG {
    std::vector<BasicBlock> blocks;  ///< All basic blocks
    int entry_block_id;               ///< Entry block ID
    int exit_block_id;                ///< Exit block ID
    
    BasicBlock* find_block_by_pc(int pc);
    const BasicBlock* find_block_by_pc(int pc) const;
    BasicBlock* find_block_by_id(int id);
    const BasicBlock* find_block_by_id(int id) const;
    void print() const;
};

/**
 * @brief Post-dominator map: PC -> immediate post-dominator PC
 */
using PostDominatorMap = std::map<int, int>;

/**
 * @brief CFG Builder - Constructs CFG and computes post-dominators
 */
class CFGBuilder {
public:
    /**
     * @brief Build CFG from kernel statements
     * @param statements Kernel statements
     * @param label2pc Label name to PC mapping
     * @return CFG Control flow graph
     */
    static CFG build(
        const std::vector<ptxemu::ir::StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    /**
     * @brief Compute post-dominators for all PCs
     * @param cfg Control flow graph
     * @return PostDominatorMap PC -> reconvergence PC
     */
    static PostDominatorMap computePostDominators(const CFG& cfg);
    
private:
    static std::vector<BasicBlock> identifyBasicBlocks(
        const std::vector<ptxemu::ir::StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    static std::set<int> findBranchTargets(
        const std::vector<ptxemu::ir::StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    static void buildEdges(CFG& cfg,
                          const std::map<std::string, int>& label2pc,
                          const std::vector<ptxemu::ir::StatementContext>& statements);
    
    static int findImmediatePostDominator(
        const BasicBlock& block,
        const std::map<int, std::set<int>>& postDomSets);
};

}
}
