/**
 * @file ptx_lane_printer.cpp
 * @brief PTX kernel SIMT 执行 + 32 位 active mask 追踪
 *
 * SIMT 执行模式：
 *   逐 PC 执行，每步记录一个 32-bit active mask（位 i=1 表示 lane i 执行了该 PC）
 *   执行完毕后通过 (PC, active_mask) 序列提取唯一执行路径
 *
 * 用法: ptx_lane_printer <ptx_file> [kernel_name] [-l]
 *   -l: 打印每 lane 完整 PC 序列（默认只打印前 4 + 摘要）
 */
#include "ptxLexer.h"
#include "ptxParser.h"
#include "ptx_parser/ptx_visiter.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/thread_state.h"
#include "ptx_ir/ptx_context.h"
#include "ptx_ir/kernel_context.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "memory/resource_manager.h"
#include "memory/simple_memory.h"
#include "memory/hardware_memory_manager.h"
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <set>

using namespace antlr4;
using namespace ptxparser;
using namespace ptxsim;

// ============================================================================
// SIMT 执行追踪结构
// ============================================================================

/// SIMT 执行步骤：(PC, active_mask)
struct SimtStep {
    int pc = 0;
    uint32_t active_mask = 0; // 位 i=1 表示 lane i 在此 PC 执行
};

using SimtTrace = std::vector<SimtStep>;

/// 一条执行路径
struct PathInfo {
    std::vector<int> lanes;        // 路径包含的 lane 编号
    std::vector<int> pcs;          // PC 序列
    bool has_branch = false;       // 路径中是否包含分支指令
};

// ============================================================================
// 辅助函数
// ============================================================================

static std::string load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static PtxContext parse_ptx(const std::string& code) {
    ANTLRInputStream input(code);
    ptxLexer lexer(&input);
    CommonTokenStream tokens(&lexer);
    tokens.fill();
    ptxParser parser(&tokens);
    PtxContext ctx;
    PtxVisitor visitor(ctx);
    visitor.visit(parser.ptxFile());
    return ctx;
}

/// 直接使用全部 kernel statements（不含过滤）
/// 声明/标签由对应 handler 自动跳过，保留完整 PC 索引
static std::vector<StatementContext> extract_executable(
    const KernelContext& kernel) {
    return kernel.kernelStatements; // 直接复制，保持 PC 一致
}

/// 从 statements 构建 label2pc（基于 S_LABEL 类型）
/// 注意：labelName 可能含 '$' 前缀，但分支指令的 target 会先 strip '$'，
/// 所以统一存储不带 '$' 的键名
static std::map<std::string, int> build_label2pc(
    const std::vector<StatementContext>& stmts) {
    std::map<std::string, int> label2pc;
    for (size_t i = 0; i < stmts.size(); i++) {
        if (stmts[i].type == S_LABEL) {
            const auto& label = std::get<LabelInstr>(stmts[i].data);
            std::string name = label.labelName;
            // strip '$' prefix if present (分支指令已 strip)
            if (!name.empty() && name[0] == '$')
                name = name.substr(1);
            label2pc[name] = static_cast<int>(i);
        }
    }
    return label2pc;
}

/// 检查指令类型是否为分支
static bool is_branch_type(StatementType type) {
    return type == S_BRA || type == S_BRX;
}

/// 检查指令类型是否为 ret/exit
static bool is_ret_type(StatementType type) {
    return type == S_RET || type == S_EXIT;
}

// ============================================================================
// 从 SIMT Trace 提取执行路径
// ============================================================================

/// 从 SIMT trace 中提取每条 lane 的 PC 序列
static std::vector<std::vector<int>> extract_lane_pc_sequences(
    const SimtTrace& trace) {
    std::vector<std::vector<int>> lane_pcs(32);
    for (const auto& step : trace) {
        uint32_t mask = step.active_mask;
        for (int i = 0; i < 32; i++) {
            if (mask & (1u << i)) {
                lane_pcs[i].push_back(step.pc);
            }
        }
    }
    return lane_pcs;
}

/// 从 lane PC 序列中提取唯一路径
static std::vector<PathInfo> extract_paths(
    const std::vector<std::vector<int>>& lane_pcs,
    const std::vector<StatementContext>& stmts) {
    std::vector<PathInfo> paths;
    std::map<std::vector<int>, int> path_map; // pc_seq → path_id
    std::vector<int> path_lane_count;

    for (int lane = 0; lane < 32; lane++) {
        const auto& seq = lane_pcs[lane];
        auto it = path_map.find(seq);
        int path_id;
        if (it == path_map.end()) {
            path_id = paths.size();
            path_map[seq] = path_id;
            paths.emplace_back();
            path_lane_count.push_back(0);
        } else {
            path_id = it->second;
        }
        paths[path_id].lanes.push_back(lane);
        paths[path_id].pcs = seq;
        path_lane_count[path_id]++;

        // 检查路径中是否包含分支指令
        for (int pc : seq) {
            if (pc >= 0 && pc < (int)stmts.size() && is_branch_type(stmts[pc].type)) {
                paths[path_id].has_branch = true;
                break;
            }
        }
    }

    return paths;
}

// ============================================================================
// SIMT 执行
// ============================================================================

/// SIMT 执行：逐 PC 执行，记录每步的 active mask
static SimtTrace simt_execute(WarpContext* warp,
                               const std::vector<StatementContext>& exec_stmts,
                               bool verbose = false) {
    SimtTrace trace;
    int num_stmts = (int)exec_stmts.size();
    static constexpr int MAX_PASSES = 1000;

    for (int pass = 0; pass < MAX_PASSES; pass++) {
        bool any_executed = false;

        // 扫描所有 PC
        for (int pc = 0; pc < num_stmts; pc++) {
            // 计算在此 PC 的 active lane mask
            uint32_t mask = 0;
            auto& wst = warp->get_warp_state();
            for (int i = 0; i < 32; i++) {
                if (!wst.threads[i].is_active || wst.threads[i].is_exited)
                    continue;
                if ((int)wst.threads[i].pc == pc)
                    mask |= (1u << i);
            }

            if (mask == 0) continue;
            any_executed = true;

            // 记录 SIMT 步骤
            trace.push_back({pc, mask});

            // 执行
            try {
                warp->execute_warp_instruction(
                    const_cast<StatementContext&>(exec_stmts[pc]), pc);
            } catch (const std::exception& e) {
                if (verbose) {
                    std::cout << "  [SIMT] PC=" << pc << " error: "
                              << e.what() << "\n";
                }
                // 执行失败的 lane 标记为 EXIT
                for (int i = 0; i < 32; i++) {
                    if (mask & (1u << i)) {
                        wst.threads[i].is_exited = true;
                        wst.threads[i].is_active = false;
                        wst.threads[i].status = ThreadStatus::Exited;
                    }
                }
            }
        }

        // 检查是否还有活跃 lane
        bool has_active = false;
        auto& wst = warp->get_warp_state();
        for (int i = 0; i < 32; i++) {
            if (wst.threads[i].is_active && !wst.threads[i].is_exited) {
                if ((int)wst.threads[i].pc < num_stmts) {
                    has_active = true;
                    break;
                }
            }
        }

        if (verbose) {
            std::cout << "  [SIMT] pass=" << pass
                      << " executed=" << (any_executed ? "Y" : "N")
                      << " active=" << (has_active ? "Y" : "N")
                      << " trace.size=" << trace.size() << "\n";
        }

        if (!has_active) break;

        if (!any_executed) break;
    }

    return trace;
}

// ============================================================================
// 打印
// ============================================================================

static void print_trace(const SimtTrace& trace,
                         const std::vector<StatementContext>& exec_stmts,
                         bool verbose) {
    if (trace.empty()) {
        std::cout << "  (empty trace)\n";
        return;
    }

    // ---- 指令统计 ----
    int branch_count = 0, ret_count = 0, label_count = 0;
    for (const auto& step : trace) {
        auto type = exec_stmts[step.pc].type;
        if (is_branch_type(type)) branch_count++;
        if (is_ret_type(type)) ret_count++;
        if (type == S_LABEL) label_count++;
    }
    std::cout << "  Execution steps: " << trace.size()
              << " (branches:" << branch_count
              << " ret:" << ret_count
              << " labels:" << label_count << ")\n";

    // ---- Lane instruction counts ----
    auto lane_pcs = extract_lane_pc_sequences(trace);
    std::cout << "  Lane instruction counts:\n";
    for (int i = 0; i < 32; i++) {
        if (i == 0 || i == 16) std::cout << "  ";
        std::cout << "L" << std::setw(2) << std::left << i << ":"
                  << std::setw(3) << std::right << lane_pcs[i].size() << " ";
        if (i == 15) std::cout << "\n";
    }
    std::cout << "\n";

    // ---- SIMT Trace（verbose 时展示 active_mask）----
    if (verbose) {
        std::cout << "  SIMT Trace (" << trace.size() << " steps):\n";
        for (size_t si = 0; si < trace.size(); si++) {
            const auto& step = trace[si];
            uint32_t m = step.active_mask;
            std::string label = (step.pc < (int)exec_stmts.size())
                ? exec_stmts[step.pc].instructionText : "???";
            if (label.length() > 40) label = label.substr(0, 40);
            std::cout << "    [" << std::setw(3) << si << "] PC=" << std::setw(2) << step.pc
                      << " mask=0x" << std::hex << std::setw(8) << std::setfill('0') << m
                      << std::dec << std::setfill(' ')
                      << "  " << label << "\n";
        }
    }

    // ---- 打印 lane trace ----
    int print_lanes = verbose ? 32 : 4;
    for (int lane = 0; lane < print_lanes && lane < 32; lane++) {
        std::cout << "  Lane " << lane << " (" << lane_pcs[lane].size()
                  << " instrs):\n    PC:";
        int count = 0;
        for (int pc : lane_pcs[lane]) {
            if (count++ > 0) std::cout << " ";
            std::cout << pc;
        }
        std::cout << "\n";
    }
    if (!verbose && 32 > print_lanes) {
        std::cout << "  ... (" << (32 - print_lanes)
                  << " more lanes, use -l to show all)\n";
    }

    // ---- 路径摘要 ----
    auto paths = extract_paths(lane_pcs, exec_stmts);
    int total_lanes = 0;
    for (size_t pi = 0; pi < paths.size(); pi++) {
        const auto& p = paths[pi];
        total_lanes += (int)p.lanes.size();

        // 描述 lane 范围
        std::string lane_str;
        if (p.lanes.size() <= 4) {
            for (size_t li = 0; li < p.lanes.size(); li++) {
                if (li > 0) lane_str += ",";
                lane_str += std::to_string(p.lanes[li]);
            }
        } else {
            lane_str = std::to_string(p.lanes.front()) + "-"
                     + std::to_string(p.lanes.back());
        }

        std::cout << "  Path " << (pi + 1) << ": lane=[" << lane_str
                  << "]  size=" << p.pcs.size()
                  << " lanes=" << p.lanes.size();
        if (p.has_branch) std::cout << " [HAS_BRANCH]";
        std::cout << "\n";

        if (verbose && !p.pcs.empty()) {
            std::cout << "    PCs:";
            for (size_t ki = 0; ki < p.pcs.size(); ki++) {
                if (ki % 20 == 0) std::cout << "\n     ";
                std::cout << " " << std::setw(2) << p.pcs[ki];
            }
            std::cout << "\n";
        }
    }
}

// ============================================================================
// Kernel 执行主线
// ============================================================================

static void print_kernel_lane_traces(
    const std::string& ptx_path,
    const std::string& kernel_name,
    bool verbose,
    bool skip_barrier = false) {

    // 1. 解析 PTX
    std::string code = load_file(ptx_path);
    if (code.empty()) {
        std::cout << "  ERROR: Cannot read " << ptx_path << "\n";
        return;
    }
    PtxContext ptx_ctx = parse_ptx(code);

    // 查找 kernel
    KernelContext* target_kernel = nullptr;
    for (auto& k : ptx_ctx.ptxKernels) {
        if (k.kernelName == kernel_name) {
            target_kernel = &k;
            break;
        }
    }
    if (!target_kernel) {
        std::cout << "  ERROR: Kernel '" << kernel_name << "' not found\n";
        return;
    }

    std::cout << "=====================================================\n";
    std::cout << "PTX: " << ptx_path << "\n";
    std::cout << "Kernel: " << kernel_name << "\n";
    std::cout << "=====================================================\n";

    // 2. 提取可执行指令序列（保留标签）
    auto exec_stmts = extract_executable(*target_kernel);
    if (exec_stmts.empty()) {
        std::cout << "  ERROR: No executable instructions\n";
        return;
    }

    // 构建 label2pc（与 exec_stmts 使用相同索引）
    auto label2pc = build_label2pc(exec_stmts);
    if (verbose) {
        std::cout << "  Labels registered: " << label2pc.size() << "\n";
        for (const auto& [name, pc] : label2pc)
            std::cout << "    '" << name << "' → PC " << pc << "\n";
    }

    // 统计
    int branch_count = 0, barrier_count = 0, label_count = 0;
    for (const auto& stmt : exec_stmts) {
        if (is_branch_type(stmt.type)) branch_count++;
        if (stmt.type == S_BAR) barrier_count++;
        if (stmt.type == S_LABEL) label_count++;
    }

    std::cout << "  Instructions: " << exec_stmts.size()
              << " (branches:" << branch_count
              << " barriers:" << barrier_count
              << " labels:" << label_count << ")\n";
    if (branch_count > 0)
        std::cout << "  Has branch: yes\n";
    if (barrier_count > 0)
        std::cout << "  Has barrier: yes (will be removed)\n";

    if (skip_barrier && barrier_count > 0) {
        std::cout << "  WARNING: bar.sync removed from execution "
                  << "(needs multi-warp scheduler)\n";
    }

    // 3. 初始化（清理前次状态）
    ResourceManager::instance().initialize(1, 65536);
    InstructionFactory::initialize();  // 内部已有 initialized 守卫
    auto* simple_mem = new SimpleMemory(64 * 1024 * 1024);
    HardwareMemoryManager::instance().set_simple_memory(simple_mem);

    // 4. 设置 SM + CTA（传完整 statements 给 CTA 构建符号表）
    auto sm = std::make_unique<SMContext>(64, 1024, 65536, 0);
    auto block = std::make_unique<CTAContext>();
    Dim3 gridDim{1, 1, 1};
    Dim3 blockDim{32, 1, 1};
    Dim3 blockIdx{0, 0, 0};
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;

    block->init(gridDim, blockDim, blockIdx, exec_stmts, &name2Sym, label2pc);

    bool ok = false;
    try {
        ok = sm->add_block(std::move(block));
    } catch (const std::exception& e) {
        std::cout << "  ERROR: add_block failed: " << e.what() << "\n";
        return;
    } catch (...) {
        std::cout << "  ERROR: add_block failed (unknown)\n";
        return;
    }
    if (!ok) {
        std::cout << "  ERROR: add_block returned false\n";
        return;
    }

    WarpContext* warp = sm->get_warp(0);
    if (!warp) {
        std::cout << "  ERROR: get_warp failed\n";
        return;
    }

    // 6. 重置 warp 状态
    auto& wst = warp->get_warp_state();
    for (int i = 0; i < 32; i++) {
        wst.threads[i].pc = 0;
        wst.threads[i].next_pc = 1;
        wst.threads[i].is_active = true;
        wst.threads[i].is_exited = false;
        wst.threads[i].is_blocked = false;
        wst.threads[i].status = ThreadStatus::Active;
        warp->set_active_mask(i, true);

        auto* t = warp->get_thread(i);
        if (t) {
            t->set_state(RUN);
            t->set_pc(0);
        }
    }
    warp->set_active_mask(0xFFFFFFFF);

    // 7. 执行（不启用 ExecutionTracer，我们用 SIMT 追踪）
    auto trace = simt_execute(warp, exec_stmts, true);

    // 8. 打印 trace
    print_trace(trace, exec_stmts, verbose);

    // 9. 提取并打印路径
    auto lane_pcs = extract_lane_pc_sequences(trace);
    auto paths = extract_paths(lane_pcs, exec_stmts);

    std::cout << "  ---\n";
    std::cout << "  Path count: " << paths.size() << "\n";
    for (size_t pi = 0; pi < paths.size(); pi++) {
        const auto& p = paths[pi];
        std::string lane_str;
        if (p.lanes.size() <= 4) {
            for (size_t li = 0; li < p.lanes.size(); li++) {
                if (li > 0) lane_str += ",";
                lane_str += std::to_string(p.lanes[li]);
            }
        } else {
            lane_str = std::to_string(p.lanes.front()) + "-"
                     + std::to_string(p.lanes.back());
        }
        std::cout << "  Path " << (pi + 1) << ": lanes=[" << lane_str
                  << "]  steps=" << p.pcs.size()
                  << " lanes=" << p.lanes.size();
        if (p.has_branch) std::cout << " [HAS_BRANCH]";
        std::cout << "\n";
    }
}

// ============================================================================
// 预选 PTX 测试列表
// ============================================================================

struct PtxTestCase {
    const char* file;
    const char* kernel;
    const char* desc;
};

static const PtxTestCase kTestCases[] = {
    // 简单无分歧
    {"tests/ptx/dummy.1.sm_80.ptx",
     "_Z7dummy_dIiEvPT_", "最简：纯算术，无分支"},

    {"tests/ptx/dummy-float.1.sm_80.ptx",
     "_Z7dummy_dIfEvPT_", "简单：float 算术，无分支"},

    // if-else 分歧
    {"tests/ptx/test_divergence_sync_standalone.ptx",
     "_Z27test_divergence_sync_kernelIiEvPT_", "复杂分歧：reduction 路径"},

    {"tests/ptx/test_if_else.ptx",
     "test_if_else", "简单 if-else：tid.x==0 分歧"},

    // 循环
    {"tests/ptx/test_loop.ptx",
     "test_loop", "循环：递减计数 3→0"},

    // 混合
    {"tests/ptx/test_mixed.ptx",
     "test_mixed", "混合：循环内嵌 if-else 分歧"},
};

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    bool verbose = false;
    std::string target_ptx;
    std::string target_kernel;

    // 解析参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-l" || arg == "--verbose") {
            verbose = true;
        } else if (target_ptx.empty()) {
            target_ptx = arg;
        } else if (target_kernel.empty()) {
            target_kernel = arg;
        }
    }

    if (target_ptx.empty()) {
        // 无参数：运行内置测试列表
        std::cout << "PTX Lane Printer — SIMT Execution Trace\n";
        std::cout << "用法: ptx_lane_printer <ptx_file> [kernel] [-l]\n\n";
        std::cout << "内置测试用例:\n";
        for (const auto& tc : kTestCases) {
            std::cout << "  " << tc.desc << "\n";
            std::cout << "    " << tc.file << " → " << tc.kernel << "\n\n";
            print_kernel_lane_traces(tc.file, tc.kernel, false, true);
            std::cout << "\n";
        }
        return 0;
    }

    print_kernel_lane_traces(target_ptx, target_kernel, verbose, true);
    return 0;
}
