#include <iostream>
#include <fstream>
#include <string>
#include "ptx_parser/ptx_visiter.h"
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/kernel_context.h"
#include "utils/logger.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <ptx_file>" << std::endl;
        return 1;
    }
    
    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "Cannot open file: " << argv[1] << std::endl;
        return 1;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    
    PtxVisitor visitor;
    visitor.visit(content);
    
    auto& kernels = visitor.getKernels();
    if (kernels.empty()) {
        std::cerr << "No kernels found" << std::endl;
        return 1;
    }
    
    std::cout << "Kernel: " << kernels[0]->name << std::endl;
    std::cout << "Statements: " << kernels[0]->kernelStatements.size() << std::endl;
    
    // Check for S_BRA statements
    for (size_t i = 0; i < kernels[0]->kernelStatements.size(); i++) {
        const auto& stmt = kernels[0]->kernelStatements[i];
        if (stmt.type == S_BRA) {
            const auto& branch = std::get<BranchInstr>(stmt.data);
            std::cout << "  [" << i << "] S_BRA target='" << branch.target 
                      << "' target.size()=" << branch.target.size() << std::endl;
        }
    }
    
    // Build label2pc
    std::map<std::string, int> label2pc;
    for (size_t i = 0; i < kernels[0]->kernelStatements.size(); i++) {
        const auto& stmt = kernels[0]->kernelStatements[i];
        if (stmt.type == S_LABEL) {
            const auto& label = std::get<LabelInstr>(stmt.data);
            label2pc[label.labelName] = i;
            std::cout << "  Label: '" << label.labelName << "' at PC=" << i << std::endl;
        }
    }
    
    // Try CFG build
    try {
        ptx::cfg::CFG cfg = ptx::cfg::CFGBuilder::build(kernels[0]->kernelStatements, label2pc);
        std::cout << "CFG built successfully: " << cfg.blocks.size() << " blocks" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "CFG build failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
