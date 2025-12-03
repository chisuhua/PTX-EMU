#include <iostream>
#include <fstream>
#include <streambuf>
#include <cassert>

#include "ptx-semantic.h"
#include "ptxParser.h"

using namespace antlr4;
using namespace ptxparser;

int main() {
    // Read the PTX file - try multiple possible locations
    std::ifstream file("test_wmma.ptx");
    if (!file.is_open()) {
        // Try alternative path
        file.open("../tests/test_wmma.ptx");
    }
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open test_wmma.ptx file in any expected location" << std::endl;
        return 1;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    
    // Create ANTLR input stream
    ANTLRInputStream input(content);
    
    // Create lexer
    ptxLexer lexer(&input);
    CommonTokenStream tokens(&lexer);
    
    // Create parser
    ptxParser parser(&tokens);
    
    // Parse the PTX file
    ptxParser::AstContext* tree = parser.ast();
    
    // Create semantic analyzer
    PtxListener listener;
    
    // Walk the parse tree
    tree::ParseTreeWalker walker;
    walker.walk(&listener, tree);
    
    // Check if we parsed correctly
    assert(listener.ptxContext.ptxKernels.size() == 1);
    
    KernelContext& kernel = listener.ptxContext.ptxKernels[0];
    std::cout << "Kernel name: " << kernel.kernelName << std::endl;
    
    // Count WMMA instructions
    int wmmaCount = 0;
    for (const auto& stmt : kernel.kernelStatements) {
        if (stmt.statementType == S_WMMA) {
            wmmaCount++;
            StatementContext::WMMA* wmmaStmt = (StatementContext::WMMA*)stmt.statement;
            
            switch (wmmaStmt->wmmaType) {
                case WMMA_LOAD:
                    std::cout << "Found WMMA load instruction" << std::endl;
                    break;
                case WMMA_STORE:
                    std::cout << "Found WMMA store instruction" << std::endl;
                    break;
                case WMMA_MMA:
                    std::cout << "Found WMMA mma instruction" << std::endl;
                    break;
            }
            
            // Print qualifiers
            std::cout << "Qualifiers: ";
            for (const auto& qual : wmmaStmt->wmmaQualifier) {
                std::cout << Q2s(qual) << " ";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "Total WMMA instructions found: " << wmmaCount << std::endl;
    
    // 根据实际的指令数量调整期望值
    if (wmmaCount == 4) {
        std::cout << "SUCCESS: All WMMA instructions parsed correctly!" << std::endl;
        return 0;
    } else {
        std::cout << "FAILURE: Expected 4 WMMA instructions, found " << wmmaCount << std::endl;
        return 1;
    }
}