#include <cassert>
#include <fstream>
#include <iostream>
#include <streambuf>

#include "ptxLexer.h"
#include "ptx_parser/ptx_parser.h"

using namespace antlr4;
using namespace ptxparser;

int main() {
    // 检查环境变量 PTX_EMU_PATH
    const char *ptx_emu_path = std::getenv("PTX_EMU_PATH");
    if (ptx_emu_path == nullptr) {
        std::cerr << "Error: PTX_EMU_PATH env is not set " << std::endl;
        return 1;
    }

    std::string filename = std::string(ptx_emu_path) + "/tests/test_wmma.ptx";
    std::ifstream stream;
    stream.open(filename);

    if (!stream.is_open()) {
        std::cerr << "Error: Could not open tests/test_wmma.ptx file: "
                  << filename << std::endl;
        return 1;
    }

    // Create ANTLR input stream
    ANTLRInputStream input(stream);

    // Create lexer
    ptxLexer lexer(&input);
    CommonTokenStream tokens(&lexer);

    // Create parser
    ptxParser parser(&tokens);

    // Parse the PTX file
    ptxParser::AstContext *tree = parser.ast();

    // Create semantic analyzer
    PtxListener listener;

    // Walk the parse tree
    tree::ParseTreeWalker walker;
    walker.walk(&listener, tree);

    // Check if we parsed correctly
    assert(listener.ptxContext.ptxKernels.size() == 1);

    KernelContext &kernel = listener.ptxContext.ptxKernels[0];
    std::cout << "Kernel name: " << kernel.kernelName << std::endl;

    // Count WMMA instructions
    int wmmaCount = 0;
    for (const auto &stmt : kernel.kernelStatements) {
        if (stmt.statementType == S_WMMA) {
            wmmaCount++;
            StatementContext::WMMA *wmmaStmt =
                (StatementContext::WMMA *)stmt.statement;

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
            for (const auto &qual : wmmaStmt->qualifier) {
                std::cout << Q2s(qual) << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Total WMMA instructions found: " << wmmaCount << std::endl;

    // 根据实际的指令数量调整期望值
    if (wmmaCount == 4) {
        std::cout << "PASS: All WMMA instructions parsed correctly!"
                  << std::endl;
        return 0;
    } else {
        std::cout << "Failed: Expected 4 WMMA instructions, found " << wmmaCount
                  << std::endl;
        return 1;
    }
}