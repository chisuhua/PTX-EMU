#include "ptxir/ptxir_serialization.h"
#include "ptx_ir/ptxir_writer.h"
#include "ptx_ir/ptxir_reader.h"
#include "ptx_parser/ptx_visiter.h"
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/ptx_context.h"
#include "antlr4-runtime.h"
#include "ptxLexer.h"
#include "ptxParser.h"
#include <fstream>
#include <map>
#include <sstream>

std::string serialize_to_string(const std::vector<struct StatementContext>& stmts) {
    std::ostringstream oss(std::ios::binary);
    ::PtxirWriter writer(oss);
    writer.write(stmts);
    return oss.str();
}

std::vector<struct StatementContext> deserialize_from_string(const std::string& data) {
    std::istringstream iss(data, std::ios::binary);
    ::PtxirReader reader(iss);
    return reader.read();
}

bool serialize_statements(const std::vector<struct StatementContext>& stmts, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    ::PtxirWriter writer(out);
    writer.write(stmts);
    return out.good();
}

std::vector<struct StatementContext> deserialize_statements(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("PTXIR file not found: " + path);
    }
    ::PtxirReader reader(in);
    return reader.read();
}

namespace {
// Fill reconvergence_pc for all S_BRA and S_BAR from CFG post-dominators.
// Shared by generate_ptxir() (embed at write time) and load_ptxir(apply_cfg=true).
void fill_reconvergence_pc(std::vector<StatementContext>& stmts) {
    std::map<std::string, int> label2pc;
    for (int i = 0; i < static_cast<int>(stmts.size()); i++) {
        if (stmts[i].type == S_LABEL) {
            const auto& label = std::get<LabelInstr>(stmts[i].data);
            label2pc[label.labelName] = i;
        }
    }
    auto cfg = ptx::cfg::CFGBuilder::build(stmts, label2pc);
    auto postDoms = ptx::cfg::CFGBuilder::computePostDominators(cfg);
    for (int i = 0; i < static_cast<int>(stmts.size()); i++) {
        if (stmts[i].type == S_BRA) {
            auto& branch = std::get<BranchInstr>(stmts[i].data);
            auto it = postDoms.find(i);
            branch.reconvergence_pc =
                (it != postDoms.end() && it->second >= 0) ? it->second : (i + 1);
        }
        if (stmts[i].type == S_BAR) {
            auto& barrier = std::get<BarrierInstr>(stmts[i].data);
            auto it = postDoms.find(i);
            barrier.reconvergence_pc =
                (it != postDoms.end() && it->second >= 0) ? it->second : (i + 1);
        }
    }
}
}  // namespace

bool generate_ptxir(const std::string& ptx_path,
                    const std::string& ptxir_path,
                    const std::string& kernel_name) {
    try {
        // Read PTX file
        std::ifstream in(ptx_path);
        if (!in) return false;
        std::string ptx_code((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());

        // Parse with ANTLR
        antlr4::ANTLRInputStream input(ptx_code);
        ptxparser::ptxLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        tokens.fill();
        ptxparser::ptxParser parser(&tokens);

        // Visit parse tree
        PtxContext ptxContext;
        PtxVisitor visitor(ptxContext);
        visitor.visit(parser.ptxFile());

        // Find kernel by name or take first
        KernelContext* kernel = nullptr;
        if (!kernel_name.empty()) {
            for (auto& k : ptxContext.ptxKernels) {
                if (k.kernelName == kernel_name) {
                    kernel = &k;
                    break;
                }
            }
        } else if (!ptxContext.ptxKernels.empty()) {
            kernel = &ptxContext.ptxKernels[0];
        }
        if (!kernel) return false;

        // Embed CFG post-dominator results before writing (design D3)
        auto statements = kernel->kernelStatements;
        fill_reconvergence_pc(statements);

        // Serialize
        return serialize_statements(statements, ptxir_path);
    } catch (...) {
        return false;
    }
}

std::vector<struct StatementContext> load_ptxir(const std::string& ptxir_path,
                                                 bool apply_cfg) {
    auto stmts = deserialize_statements(ptxir_path);
    if (apply_cfg) {
        fill_reconvergence_pc(stmts);
    }
    return stmts;
}


