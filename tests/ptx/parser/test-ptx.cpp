/**
 * @author gtyinstinct
 * test lexer parser semantic
 */

#include "ptxLexer.h"
#include "ptxParser.h"
#include "ptxParserBaseVisitor.h"
#include "ptx_ir/ptx_context.h"
#include "ptx_parser/ptx_visiter.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace antlr4;
using namespace ptxparser;

namespace {

void dumpStatements(const std::vector<ptxemu::ir::StatementContext> &statements,
                          std::ostream &os, const std::string &indent) {
     for (size_t i = 0; i < statements.size(); ++i) {
          os << indent << "[" << i << "] " << statements[i].toString()
              << std::endl;

        if (std::holds_alternative<GenericInstr>(statements[i].data)) {
            const auto &g = std::get<GenericInstr>(statements[i].data);
            for (size_t opi = 0; opi < g.operands.size(); ++opi) {
                const auto &op = g.operands[opi];
                os << indent << "      op[" << opi << "] kind="
                   << static_cast<int>(op.kind());
                if (op.kind() == OperandKind::ADDR) {
                    const auto &addr = std::get<AddrOperand>(op.data);
                    const char *offsetType =
                        addr.offsetType == AddrOperand::OffsetType::REGISTER
                            ? "REGISTER"
                            : "IMMEDIATE";
                    os << " addr.id=" << addr.id
                       << " addr.baseSymbol=" << addr.baseSymbol
                       << " addr.offsetType=" << offsetType
                       << " addr.immediateOffset="
                       << addr.immediateOffset;
                    if (addr.registerOffset) {
                        os << " addr.registerOffset="
                           << addr.registerOffset->toString();
                    }
                } else if (op.kind() == OperandKind::VAR) {
                    const auto &var = std::get<VariableOperand>(op.data);
                    os << " var.name=" << var.name;
                } else if (op.kind() == OperandKind::REG) {
                    const auto &reg = std::get<RegOperand>(op.data);
                    os << " reg=" << reg.fullName();
                }
                os << std::endl;
            }
        }
     }
}

void dumpPtxContext(const PtxContext &ptxContext, std::ostream &os) {
     os << "=== Visitor Dump ===" << std::endl;
     os << "PTX version: " << ptxContext.ptxMajorVersion << "."
         << ptxContext.ptxMinorVersion << std::endl;
     os << "PTX target: sm_" << ptxContext.ptxTarget << std::endl;
     os << "Address size: " << ptxContext.ptxAddressSize << std::endl;
     os << "Top-level statements: " << ptxContext.ptxStatements.size()
         << std::endl;

     if (!ptxContext.ptxStatements.empty()) {
          dumpStatements(ptxContext.ptxStatements, os, "  ");
     }

     os << "Kernels: " << ptxContext.ptxKernels.size() << std::endl;
     for (size_t kernelIndex = 0; kernelIndex < ptxContext.ptxKernels.size();
            ++kernelIndex) {
          const auto &kernel = ptxContext.ptxKernels[kernelIndex];
          os << "Kernel[" << kernelIndex << "]" << std::endl;
          os << "  name: " << kernel.kernelName << std::endl;
          os << "  visible: " << std::boolalpha << kernel.ifVisibleKernel
              << std::endl;
          os << "  entry: " << std::boolalpha << kernel.ifEntryKernel
              << std::endl;
          os << "  params: " << kernel.kernelParams.size() << std::endl;
          os << "  statements: " << kernel.kernelStatements.size()
              << std::endl;
          dumpStatements(kernel.kernelStatements, os, "    ");
     }

     os << "Extern functions: " << ptxContext.externFuncs.size() << std::endl;
}

void dumpParamAndRegFromTree(ptxparser::ptxParser::PtxFileContext *tree,
                             std::ostream &os) {
    if (tree == nullptr) {
        return;
    }

    os << "=== Parse Tree Param/Reg Dump ===" << std::endl;
    
    // Process function declarations from ptxFile directly
    for (auto *funcDecl : tree->functionDecl()) {
        if (!funcDecl) {
            continue;
        }
        
        auto *funcHeader = funcDecl->functionHeader();
        auto *funcBody = funcDecl->funcBody();

        std::string kernelName = "<unknown>";
        if (funcHeader && funcHeader->ID()) {
            kernelName = funcHeader->ID()->getText();
        }
        os << "Kernel: " << kernelName << std::endl;

        size_t paramCount = 0;
        if (funcHeader && funcHeader->paramList()) {
            const auto &params = funcHeader->paramList()->paramDecl();
            paramCount = params.size();
            os << "  params: " << paramCount << std::endl;
            for (size_t i = 0; i < params.size(); ++i) {
                os << "    [" << i << "] " << params[i]->getText() << std::endl;
            }
        } else {
            os << "  params: 0" << std::endl;
        }

        size_t regCount = 0;
        if (funcBody) {
            const auto &regDecls = funcBody->regDecl();
            regCount = regDecls.size();
            os << "  regDecls: " << regCount << std::endl;
            for (size_t i = 0; i < regDecls.size(); ++i) {
                os << "    [" << i << "] " << regDecls[i]->getText()
                   << std::endl;
            }
        } else {
            os << "  regDecls: 0" << std::endl;
        }
    }
}

} // namespace

int main(int argc, const char *argv[]) {
    std::string filename;
    if (argc >= 2) {
        filename = argv[1];
    } else {
        const char *ptx_emu_path = std::getenv("PTX_EMU_PATH");
        if (ptx_emu_path == nullptr) {
            std::cerr << "Warning: PTX_EMU_PATH not set; "
                      << "using default path ./tests/ptx/atom_cas_basic.ptx"
                      << std::endl;
            filename = "./tests/ptx/atom_cas_basic.ptx";
        } else {
            filename = std::string(ptx_emu_path) + "/tests/ptx/atom_cas_basic.ptx";
        }
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

    ptxParser parser(&tokens);

    ptxParser::PtxFileContext *tree = parser.ptxFile();

    PtxContext ptxContext;
    PtxVisitor visitor(ptxContext);
    visitor.visit(tree);

    dumpPtxContext(ptxContext, std::cout);
    dumpParamAndRegFromTree(tree, std::cout);
    std::cout << std::endl;

    if (tree != nullptr && parser.getNumberOfSyntaxErrors() == 0) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }
    stream.close();

    return 0;
}
