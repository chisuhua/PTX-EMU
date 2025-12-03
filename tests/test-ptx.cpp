/**
 * @author gtyinstinct
 * test lexer parser semantic
 */

#include "ptx-semantic.h"

// #define TOKEN
// #define TREE
#define SEMANTIC

int main(int argc, const char *argv[])
{
  // 如果没有提供参数，使用默认的测试文件
  std::string filename = "test_wmma.ptx";
  if (argc >= 2)
  {
    filename = argv[1];
  }

  std::ifstream stream;
  stream.open(filename);

  // 如果在当前目录找不到，尝试在上一级目录的bench文件夹中查找
  if (!stream.is_open())
  {
    std::string alt_filename = "../tests/test_wmma.ptx";
    if (argc >= 2)
    {
      alt_filename = std::string("../") + argv[1];
    }
    stream.open(alt_filename);
  }

  if (!stream.is_open())
  {
    std::cerr << "Error: Could not open PTX file: " << filename << std::endl;
    return 1;
  }

  ANTLRInputStream input(stream);

  ptxLexer lexer(&input);
  CommonTokenStream tokens(&lexer);

  tokens.fill();

#ifdef TOKEN
  // output tokens
  for (auto token : tokens.getTokens())
  {
    std::cout << token->toString() << std::endl;
  }
#endif

  ptxParser parser(&tokens);
  PtxListener tl;
  parser.addParseListener(&tl);

  tree::ParseTree *tree = parser.ast();

#ifdef TREE
  // output grammar tree
  std::cout << tree->toStringTree(&parser) << std::endl
            << std::endl;
#endif

#ifdef SEMANTIC
  // output semantic
  tl.test_semantic();
#endif

  return 0;
}