#include "cfg_builder.h"
#include "ptx_ir/statement_context.h"
#include <algorithm>
#include <iostream>

namespace ptx {
namespace cfg {

BasicBlock* CFG::find_block_by_pc(int pc) {
    for (auto& block : blocks) {
        if (block.contains(pc)) {
            return &block;
        }
    }
    return nullptr;
}

BasicBlock* CFG::find_block_by_id(int id) {
    for (auto& block : blocks) {
        if (block.id == id) {
            return &block;
        }
    }
    return nullptr;
}

void CFG::print() const {
    std::cout << "CFG: " << blocks.size() << " blocks, entry=" 
              << entry_block_id << ", exit=" << exit_block_id << std::endl;
    for (const auto& block : blocks) {
        std::cout << "  Block " << block.id << " [PC=" << block.start_pc 
                  << "-" << block.end_pc << "]";
        if (block.is_branch_target) std::cout << " (branch target)";
        if (block.is_exit) std::cout << " (exit)";
        std::cout << std::endl;
    }
}

std::set<int> CFGBuilder::findBranchTargets(
    const std::vector<StatementContext>& statements,
    const std::map<std::string, int>& label2pc) {
    
    std::set<int> targets;
    
    for (const auto& stmt : statements) {
        if (stmt.type == S_BRA) {
            const auto& branch = std::get<BranchInstr>(stmt.data);
            // Skip branches with empty target (malformed or incomplete PTX)
            if (branch.target.empty()) {
                continue;
            }
            auto it = label2pc.find(branch.target);
            if (it != label2pc.end()) {
                targets.insert(it->second);
            } else {
                std::cerr << "[CFGBuilder] Warning: Branch target '" 
                          << branch.target << "' not found" << std::endl;
            }
        }
    }
    
    return targets;
}

std::vector<BasicBlock> CFGBuilder::identifyBasicBlocks(
    const std::vector<StatementContext>& statements,
    const std::map<std::string, int>& label2pc) {
    
    std::set<int> boundaries;
    boundaries.insert(0);
    boundaries.insert(statements.size());
    
    auto targets = findBranchTargets(statements, label2pc);
    boundaries.insert(targets.begin(), targets.end());
    
    for (size_t i = 0; i < statements.size(); i++) {
        if (statements[i].type == S_BRA) {
            boundaries.insert(i + 1);
        }
    }
    
    std::vector<BasicBlock> blocks;
    int block_id = 0;
    int prev_boundary = 0;
    
    for (int boundary : boundaries) {
        if (boundary > prev_boundary) {
            BasicBlock block;
            block.id = block_id++;
            block.start_pc = prev_boundary;
            block.end_pc = boundary;
            block.is_branch_target = (targets.count(prev_boundary) > 0);
            block.is_exit = false;
            
            blocks.push_back(block);
        }
        prev_boundary = boundary;
    }
    
    return blocks;
}

void CFGBuilder::buildEdges(CFG& cfg,
                            const std::map<std::string, int>& label2pc,
                            const std::vector<StatementContext>& statements) {
    
    for (size_t i = 0; i < cfg.blocks.size(); i++) {
        BasicBlock& block = cfg.blocks[i];
        
        if (block.is_exit) continue;
        
        int last_pc = block.end_pc - 1;
        if (last_pc < 0 || last_pc >= (int)statements.size()) continue;
        
        const auto& stmt = statements[last_pc];
        
        if (stmt.type == S_BRA) {
            const auto& branch = std::get<BranchInstr>(stmt.data);
            
            // 1. Add fall-through edge
            for (auto& other : cfg.blocks) {
                if (other.start_pc == block.end_pc) {
                    bool found = false;
                    for (int succ : block.successors) {
                        if (succ == other.id) { found = true; break; }
                    }
                    if (!found) {
                        block.successors.push_back(other.id);
                        other.predecessors.push_back(block.id);
                    }
                }
            }
            
            // 2. Add branch target edge
            int target_pc = -1;
            auto it = label2pc.find(branch.target);
            if (it != label2pc.end()) {
                target_pc = it->second;
            }
            
            if (target_pc >= 0) {
                for (auto& other : cfg.blocks) {
                    if (other.start_pc == target_pc) {
                        bool found = false;
                        for (int succ : block.successors) {
                            if (succ == other.id) { found = true; break; }
                        }
                        if (!found) {
                            block.successors.push_back(other.id);
                            other.predecessors.push_back(block.id);
                        }
                    }
                }
            }
        } else {
            for (auto& other : cfg.blocks) {
                if (other.start_pc == block.end_pc) {
                    block.successors.push_back(other.id);
                    other.predecessors.push_back(block.id);
                    break;
                }
            }
        }
    }
}

CFG CFGBuilder::build(
    const std::vector<StatementContext>& statements,
    const std::map<std::string, int>& label2pc) {
    
    CFG cfg;
    cfg.blocks = identifyBasicBlocks(statements, label2pc);
    cfg.entry_block_id = 0;
    cfg.exit_block_id = cfg.blocks.empty() ? 0 : cfg.blocks.size() - 1;
    
    if (!cfg.blocks.empty()) {
        cfg.blocks.back().is_exit = true;
    }
    
    buildEdges(cfg, label2pc, statements);
    
    return cfg;
}

PostDominatorMap CFGBuilder::computePostDominators(const CFG& cfg) {
    std::map<int, std::set<int>> postDomSets;
    PostDominatorMap result;
    
    std::set<int> all_block_ids;
    for (const auto& block : cfg.blocks) {
        all_block_ids.insert(block.id);
    }
    
    for (const auto& block : cfg.blocks) {
        if (block.id == cfg.exit_block_id) {
            postDomSets[block.id] = {block.id};
        } else {
            postDomSets[block.id] = all_block_ids;
        }
    }
    
    bool changed = true;
    int iterations = 0;
    while (changed && iterations < 100) {
        changed = false;
        iterations++;
        
        for (const auto& block : cfg.blocks) {
            if (block.id == cfg.exit_block_id) continue;
            
            // 标准后支配集算法：用全量集合初始化，求后继交集，最后加入自身
            std::set<int> newSet = all_block_ids;
            
            if (!block.successors.empty()) {
                for (int succ_id : block.successors) {
                    auto it = postDomSets.find(succ_id);
                    if (it == postDomSets.end()) continue;
                    
                    std::set<int> intersection;
                    std::set_intersection(
                        newSet.begin(), newSet.end(),
                        it->second.begin(), it->second.end(),
                        std::inserter(intersection, intersection.begin())
                    );
                    newSet = intersection;
                }
            }
            newSet.insert(block.id);
            
            if (newSet != postDomSets[block.id]) {
                postDomSets[block.id] = newSet;
                changed = true;
            }
        }
    }
    
    // Build a map from block ID to start_pc for post-dominator resolution
    std::map<int, int> blockIdToPC;
    for (const auto& block : cfg.blocks) {
        blockIdToPC[block.id] = block.start_pc;
    }
    
    for (const auto& block : cfg.blocks) {
        int ipd_block_id = findImmediatePostDominator(block, postDomSets);
        int postDomPC;
        if (ipd_block_id >= 0) {
            auto it2 = blockIdToPC.find(ipd_block_id);
            if (it2 != blockIdToPC.end()) {
                postDomPC = it2->second;
            } else {
                postDomPC = block.end_pc;
            }
        } else {
            postDomPC = block.end_pc;
        }
        for (int pc = block.start_pc; pc < block.end_pc; pc++) {
            result[pc] = postDomPC;
        }
    }
    
    return result;
}

int CFGBuilder::findImmediatePostDominator(
    const BasicBlock& block,
    const std::map<int, std::set<int>>& postDomSets) {
    
    auto it = postDomSets.find(block.id);
    if (it == postDomSets.end()) return -1;
    
    const std::set<int>& postDoms = it->second;
    
    for (int candidate : postDoms) {
        if (candidate == block.id) continue;
        
        bool isImmediate = true;
        for (int other : postDoms) {
            if (other == block.id || other == candidate) continue;
            
            auto otherIt = postDomSets.find(other);
            if (otherIt == postDomSets.end()) continue;
            
            if (otherIt->second.count(candidate)) {
                isImmediate = false;
                break;
            }
        }
        
        if (isImmediate) {
            return candidate;
        }
    }
    
    return -1;
}

}
}
