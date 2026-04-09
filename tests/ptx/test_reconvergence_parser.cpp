/**
 * Test reconvergence_pc computation during PTX parsing (SIMT v2.0)
 * 
 * This test verifies that:
 * 1. CFG Builder correctly identifies branch targets
 * 2. Post-Dominator analysis computes correct reconvergence points
 * 3. BranchInstr.reconvergence_pc is correctly populated
 */

#include "ptx_parser/ptx_parser.h"
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/kernel_context.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace ptx;

bool load_ptx_file(const std::string& filename, std::string& content) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << std::endl;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    content = buffer.str();
    return true;
}

int check_reconvergence_pc(KernelContext* kernel_ctx, const std::string& kernel_name) {
    std::cout << "\n=== Checking " << kernel_name << " ===" << std::endl;
    
    int total_branches = 0;
    int branches_with_reconvergence = 0;
    int branches_with_invalid_reconvergence = 0;
    
    for (int i = 0; i < kernel_ctx->kernelStatements.size(); i++) {
        const auto& stmt = kernel_ctx->kernelStatements[i];
        
        if (stmt.type == S_BRA) {
            total_branches++;
            const auto& branch = std::get<BranchInstr>(stmt.data);
            
            std::cout << "PC=" << std::to_string(i) 
                      << ": bra " << branch.target
                      << ", reconvergence_pc=" << std::to_string(branch.reconvergence_pc);
            
            if (branch.reconvergence_pc < 0) {
                std::cout << " ❌ FAIL (reconvergence_pc not set)" << std::endl;
                branches_with_invalid_reconvergence++;
            } else if (branch.reconvergence_pc <= i) {
                std::cout << " ❌ FAIL (reconvergence_pc <= branch_pc)" << std::endl;
                branches_with_invalid_reconvergence++;
            } else {
                std::cout << " ✅ PASS" << std::endl;
                branches_with_reconvergence++;
            }
        }
    }
    
    std::cout << "--- Summary for " << kernel_name << " ---" << std::endl;
    std::cout << "Total branches: " << total_branches << std::endl;
    std::cout << "With reconvergence_pc: " << branches_with_reconvergence << std::endl;
    std::cout << "Invalid reconvergence_pc: " << branches_with_invalid_reconvergence << std::endl;
    
    return (total_branches > 0 && branches_with_invalid_reconvergence == 0) ? 0 : 1;
}

int test_cfg_builder(KernelContext* kernel_ctx) {
    std::cout << "\n=== Running CFG Builder Test ===" << std::endl;
    
    try {
        // Build label2pc map
        std::map<std::string, int> label2pc;
        for (int i = 0; i < kernel_ctx->kernelStatements.size(); i++) {
            const auto& stmt = kernel_ctx->kernelStatements[i];
            if (stmt.type == S_DOLLOR) {
                const auto& dollar = std::get<DollarNameInstr>(stmt.data);
                label2pc[dollar.name] = i;
            }
        }
        
        std::cout << "Labels found: " << label2pc.size() << std::endl;
        for (const auto& [name, pc] : label2pc) {
            std::cout << "  " << name << " -> PC=" << pc << std::endl;
        }
        
        // Run CFG analysis
        std::cout << "\nRunning CFG Builder..." << std::endl;
        ptx::cfg::CFG cfg = ptx::cfg::CFGBuilder::build(kernel_ctx->kernelStatements, label2pc);
        
        std::cout << "CFG built: " << cfg.blocks.size() << " basic blocks" << std::endl;
        
        std::cout << "Running Post-Dominator analysis..." << std::endl;
        ptx::cfg::PostDominatorMap postDoms = ptx::cfg::CFGBuilder::computePostDominators(cfg);
        
        std::cout << "Post-Dominators computed for " << postDoms.size() << " PCs" << std::endl;
        
        // Update BranchInstr with reconvergence_pc
        int updated = 0;
        for (int i = 0; i < kernel_ctx->kernelStatements.size(); i++) {
            auto& stmt = kernel_ctx->kernelStatements[i];
            if (stmt.type == S_BRA) {
                auto& branch = std::get<BranchInstr>(stmt.data);
                
                auto it = postDoms.find(i);
                if (it != postDoms.end() && it->second >= 0) {
                    branch.reconvergence_pc = it->second;
                    updated++;
                } else {
                    branch.reconvergence_pc = i + 1;
                }
            }
        }
        
        std::cout << "Updated " << updated << " branch instructions with reconvergence_pc" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CFG analysis failed: " << e.what() << std::endl;
        return 1;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== SIMT v2.0 Reconvergence PC Parser Test ===" << std::endl;
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <ptx_file>" << std::endl;
        return 1;
    }
    
    std::string ptx_filename = argv[1];
    std::cout << "Loading PTX file: " << ptx_filename << std::endl;
    
    std::string ptx_code;
    if (!load_ptx_file(ptx_filename, ptx_code)) {
        return 1;
    }
    
    std::cout << "PTX code loaded: " << ptx_code.size() << " bytes" << std::endl;
    
    // Create parser
    PtxParser parser;
    
    std::cout << "Parsing PTX..." << std::endl;
    if (!parser.parse(ptx_code)) {
        std::cerr << "Parser failed!" << std::endl;
        return 1;
    }
    
    std::cout << "Parsing successful!" << std::endl;
    std::cout << "Kernels found: " << parser.kernel_contexts.size() << std::endl;
    
    int failures = 0;
    
    // Test each kernel
    for (auto& [name, kernel_ctx] : parser.kernel_contexts) {
        // First run CFG builder to populate reconvergence_pc
        if (test_cfg_builder(kernel_ctx.get()) != 0) {
            std::cerr << "CFG Builder test failed for " << name << std::endl;
            failures++;
            continue;
        }
        
        // Then check reconvergence_pc
        if (check_reconvergence_pc(kernel_ctx.get(), name) != 0) {
            std::cerr << "Reconvergence PC check failed for " << name << std::endl;
            failures++;
        }
    }
    
    std::cout << "\n=== Overall Result ===" << std::endl;
    if (failures == 0) {
        std::cout << "✅ ALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "❌ " << failures << " test(s) FAILED" << std::endl;
        return 1;
    }
}
