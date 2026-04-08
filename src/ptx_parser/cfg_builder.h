// ============================================================================
// PTX-EMU SIMT v2.0 - CFG (Control Flow Graph) Builder
// ============================================================================
// File: cfg_builder.h
// Description: Builds Control Flow Graph from PTX kernel statements and computes
//              post-dominators for SIMT reconvergence analysis
// Author: PTX-EMU Architecture Team
// Date: 2026-04-09
// Version: 2.0
// ============================================================================

#pragma once

#include <vector>
#include <map>
#include <set>
#include <string>
#include <optional>

#include "ptx_ir/statement_context.h"
#include "ptx_ir/symtable.h"

namespace ptx {
namespace cfg {

/**
 * @brief represents a basic block in the control flow graph
 * 
 * A basic block is a maximal sequence of consecutive statements
 * with: single entry (first instruction) and 
 * single exit (last instruction)
 */
struct BasicBlock {
    int id;                         // Unique block identifier
    int start_pc;                   // Starting program counter (inclusive)
    int end_pc;                     // Ending program counter (exclusive)
    std::string label_name;         // Label name if block starts with label
    
    std::vector<int> successors;    // IDs of successor blocks
    std::vector<int> predecessors;  // IDs of predecessor blocks
    
    bool is_branch_target;          // Is this block a branch target?
    bool is_exit;                   // Is this the exit block?
    
    /**
     * @brief Get the number of statements in this block
     */
    int size() const { return end_pc - start_pc; }
    
    /**
     * @brief Check if a PC is within this block
     */
    bool contains(int pc) const { 
        return pc >= start_pc && pc < end_pc; 
    }
    
    /**
     * @brief Check if this block is empty (size 0)
     */
    bool empty() const { return size() == 0; }
};

/**
 * @brief Control Flow Graph representation
 */
struct CFG {
    std::vector<BasicBlock> blocks;    // All basic blocks
    int entry_block_id;                 // Entry block ID
    int exit_block_id;                  // Exit block ID
    
    /**
     * @brief Find a block by its program counter
     */
    BasicBlock* find_block_by_pc(int pc);
    const BasicBlock* find_block_by_pc(int pc) const;
    
    /**
     * @brief Find a block by its ID
     */
    BasicBlock* find_block_by_id(int id);
    const BasicBlock* find_block_by_id(int id) const;
    
    /**
     * @brief Print CFG (for debugging)
     */
    void print() const;
};

/**
 * @brief Post-dominator computation result
 * 
 * Maps each PC to its immediate post-dominator PC.
 * If a PC has no post-dominator (exit), maps to -1.
 */
using PostDominatorMap = std::map<int, int>;

/**
 * @brief CFG Builder - Builds control flow graph and computes post-dominators
 * 
 * This class implements the following algorithms:
 * 1. Basic Block Identification: Partitions statements into basic blocks
 * 2. CFG Construction: Builds successor/predecessor edges
 * 3. Post-Dominator Computation: Iterative algorithm to find post-dominators
 * 
 * Reference: Cytron et al. "Simple and Efficient Construction of Static 
 *            Single Assignment Forms with Optimal Dominator Frontiers"
 */
class CFGBuilder {
public:
    /**
     * @brief Build CFG from kernel statements
     * 
     * @param statements Kernel statements (from parser)
     * @param label2pc Label name to PC mapping
     * @return CFG Control flow graph
     */
    static CFG build(
        const std::vector<StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    /**
     * @brief Compute post-dominators for all PCs in the CFG
     * 
     * @param cfg Control flow graph
     * @return PostDominatorMap PC -> immediate post-dominator PC
     */
    static PostDominatorMap computePostDominators(const CFG& cfg);
    
    /**
     * @brief Compute post-dominators directly from statements (convenience)
     * 
     * @param statements Kernel statements
     * @param label2pc Label name to PC mapping
     * @return PostDominatorMap PC -> immediate post-dominator PC
     */
    static PostDominatorMap computePostDominators(
        const std::vector<StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
private:
    /**
     * @brief Identify basic blocks from statements
     * 
     * Finds block boundaries at:
     * - Entry (PC=0)
     * - Exit (PC=statements.size())
     * - Branch targets
     * - Instructions following branches
     * 
     * @param statements Kernel statements
     * @param label2pc Label name to PC mapping
     * @return std::vector<BasicBlock> Identified basic blocks
     */
    static std::vector<BasicBlock> identifyBasicBlocks(
        const std::vector<StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    /**
     * @brief Find all branch targets in the kernel
     * 
     * @param statements Kernel statements
     * @param label2pc Label name to PC mapping
     * @return std::set<int> Set of PCs that are branch targets
     */
    static std::set<int> findBranchTargets(
        const std::vector<StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    /**
     * @brief Build successor and predecessor edges for CFG
     * 
     * For each block:
     * - Non-branch: successor is next block
     * - Branch: successors are target block and fall-through block
     * 
     * @param cfg CFG to build edges for (modified in-place)
     * @param statements Kernel statements
     */
    static void buildEdges(
        CFG& cfg,
        const std::vector<StatementContext>& statements);
    
    /**
     * @brief Find immediate post-dominator for a block
     * 
     * The immediate post-dominator is the closest post-dominator
     * (first post-dominator that dominates all other post-dominators)
     * 
     * @param block Block to find post-dominator for
     * @param postDomSets Post-dominator sets for all blocks
     * @return int Immediate post-dominator PC, or -1 if none
     */
    static int findImmediatePostDominator(
        const BasicBlock& block,
        const std::map<int, std::set<int>>& postDomSets);
};

} // namespace cfg
} // namespace ptx
