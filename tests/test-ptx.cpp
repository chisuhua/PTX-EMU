/**
 * @author gtyinstinct
 * test lexer parser semantic
 */

#include "ptxLexer.h"
#include "ptxParser.h"
#include "ptxParserBaseVisitor.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace antlr4;
using namespace ptxparser;

int main(int argc, const char *argv[]) {
    std::string filename;
    if (argc >= 2) {
        filename = argv[1];
    } else {
        const char *ptx_emu_path = std::getenv("PTX_EMU_PATH");
        if (ptx_emu_path == nullptr) {
            std::cerr << "Error: PTX_EMU_PATH environment variable not set"
                      << std::endl;
            return 1;
        }
        filename = std::string(ptx_emu_path) + "/tests/ptx/dummy.1.sm_75.ptx";
    }

    std::ifstream stream;
    stream.open(filename);

    if (!stream.is_open()) {
        std::cerr << "Error: Could not open PTX file: " << filename
                  << std::endl;
        return 1;
    }

    ANTLRInputStream input(stream);

    ptxLexer lexer(&input);
    CommonTokenStream tokens(&lexer);

    tokens.fill();

#ifdef TOKEN
    for (auto token : tokens.getTokens()) {
        std::cout << token->toString() << std::endl;
    }
#endif

    ptxParser parser(&tokens);

    ptxParser::PtxFileContext *tree = parser.ptxFile();

#ifdef TREE
    std::cout << tree->toStringTree(&parser) << std::endl << std::endl;
#endif

    if (tree != nullptr) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }
    stream.close();

    return 0;
}
