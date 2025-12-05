/**
 * @author gtyinstinct
 * test lexer parser semantic
 */

// #define TOKEN
// #define TREE
#include "ptxLexer.h"
#include "ptx_parser/ptx_parser.h"
#include <ostream>
#include <string>

#define SEMANTIC

int main(int argc, const char *argv[]) {
    // 如果没有提供参数，使用默认的测试文件
    std::string filename;
    if (argc >= 2) {
        filename = argv[1];
    }

    // 检查环境变量 PTX_EMU_PATH
    const char *ptx_emu_path = std::getenv("PTX_EMU_PATH");
    if (ptx_emu_path == nullptr) {
        std::cerr << "Error: PTX_EMU_PATH environment variable not set"
                  << std::endl;
        return 1;
    }

    filename = std::string(ptx_emu_path) + "/tests/test_ptx.ptx";

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
    // output tokens
    for (auto token : tokens.getTokens()) {
        std::cout << token->toString() << std::endl;
    }
#endif

    ptxParser parser(&tokens);
    PtxListener tl;
    parser.addParseListener(&tl);

    tree::ParseTree *tree = parser.ast();

#ifdef TREE
    // output grammar tree
    std::cout << tree->toStringTree(&parser) << std::endl << std::endl;
#endif

#ifdef SEMANTIC
    // output semantic
    tl.test_semantic();
#endif

    std::cout << "PASS" << std::endl;
    stream.close();

    return 0;
}