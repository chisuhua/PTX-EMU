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
 * 
 * A basic block is a maximal sequence of consecutive statements
 * with a single entry point and single exit point.
 */
struct BasicBlock {
    int id;               ///< Block identifier
    int start_pc;         ///< Starting program counter (inclusive)
    int end_pc;           ///< Ending program counter (exclusive)
    
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
 * 
 * Represents the control flow of a PTX kernel as a directed graph
 * of basic blocks.
 * 
 * @example
 * CFG cfg = CFGBuilder::build(statements, label2pc);
 * for (const auto& block : cfg.blocks) {
 *     std::cout << "Block " << block.id 
 *               << " [PC=" << block.start_pc << "-" << block.end_pc << "]" 
 *               << std::endl;
 * }
 */
struct CFG {
    std::vector<BasicBlock> blocks;  ///< All basic blocks
    int entry_block_id;               ///< Entry block ID (usually 0)
    int exit_block_id;                ///< Exit block ID (usually last block)
    
    BasicBlock* find_block_by_pc(int pc);
    const BasicBlock* find_block_by_pc(int pc) const;
    BasicBlock* find_block_by_id(int id);
    const BasicBlock* find_block_by_id(int id) const;
    void print() const;
};

/**
 * @brief Post-dominator map: PC -> immediate post-dominator PC
 * 
 * Maps each program counter to its immediate post-dominator.
 * The immediate post-dominator is the point where all divergent paths
 * reconverge.
 * 
 * @note If a PC has no post-dominator (e.g., exit block), the value is -1.
 */
using PostDominatorMap = std::map<int, int>;

/**
 * @brief CFG Builder - Constructs CFG and computes post-dominators
 * 
 * This class provides functionality to:
 * 1. Build a control flow graph from PTX kernel statements
 * 2. Compute post-dominators for branch reconvergence analysis
 * 
 * @example
 * // Build CFG from kernel statements
 * CFG cfg = CFGBuilder::build(statements, label2pc);
 * 
 * // Compute post-dominators for reconvergence points
 * PostDominatorMap postDoms = CFGBuilder::computePostDominators(cfg);
 * 
 * // Update branch instructions with reconvergence PC
 * for (size_t i = 0; i < statements.size(); i++) {
 *     if (statements[i].type == S_BRA) {
 *         auto& branch = std::get<BranchInstr>(statements[i].data);
 *         auto it = postDoms.find(i);
 *         if (it != postDoms.end() && it->second >= 0) {
 *             branch.reconvergence_pc = it->second;
 *         } else {
 *             branch.reconvergence_pc = i + 1; // Fallback
 *         }
 *     }
 * }
 * 
 * @note The CFG builder handles various branch patterns:
 * - Simple if-else
 * - Nested branches (multiple levels)
 * - Multi-way branches (switch-case)
 * - Loop structures (while, for)
 */
class CFGBuilder {
public:
    /**
     * @brief Build CFG from kernel statements
     * 
     * Analyzes kernel statements to construct a control flow graph.
     * Identifies basic blocks and computes successor/predecessor edges.
     * 
     * @param statements Kernel statements (from PTX parser)
     * @param label2pc Label name to PC mapping
     * @return CFG Constructed control flow graph
     * 
     * @note Time complexity: O(n) where n = number of statements
     */
    static CFG build(
        const std::vector<StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    /**
     * @brief Compute post-dominators for all PCs
     * 
     * Uses iterative data-flow algorithm to compute immediate post-dominators.
     * Post-dominators represent branch reconvergence points.
     * 
     * @param cfg Control flow graph (from build())
     * @return PostDominatorMap PC -> immediate post-dominator PC
     * 
     * @note Time complexity: O(n * iterations), typically converges in < 100 iterations
     * @note Uses fixed-point iteration to ensure termination
     */
    static PostDominatorMap computePostDominators(const CFG& cfg);
    
private:
    static std::vector<BasicBlock> identifyBasicBlocks(
        const std::vector<StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    static std::set<int> findBranchTargets(
        const std::vector<StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    static void buildEdges(CFG& cfg,
                          const std::map<std::string, int>& label2pc,
                          const std::vector<StatementContext>& statements);
    
    static int findImmediatePostDominator(
        const BasicBlock& block,
        const std::map<int, std::set<int>>& postDomSets);
};

} // namespace cfg
} // namespace ptx
