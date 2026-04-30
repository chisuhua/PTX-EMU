/**
 * @file test_helpers.hpp
 * @brief Five-Mode Testing Framework - Common Helpers
 *
 * 提供三种 PTX 测试模式的公共基础设施函数。
 */

#ifndef TEST_HELPERS_HPP
#define TEST_HELPERS_HPP

#include "catch_amalgamated.hpp"
#include "ptxsim/warp_state.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/wbar.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <sstream>

// ============================================================================
// Type Aliases
// ============================================================================

// WarpContext is a global class (not in namespace)
using WarpContext = WarpContext;
using ThreadState = ptxsim::ThreadState;
using Wbar = ptxsim::Wbar;
using ThreadStatus = ptxsim::ThreadStatus;

// ============================================================================
// Instruction Factory Initialization
// ============================================================================

inline void init_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

// ============================================================================
// Statement Construction Helpers (Mode 3)
// ============================================================================

inline StatementContext make_bar_warp_sync(uint32_t mask, int reconvergence_pc) {
    StatementContext ctx;
    ctx.type = S_BAR_WARP_SYNC;
    BarWarpSyncInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(mask)}});
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(reconvergence_pc)}});
    ctx.data = instr;
    ctx.instructionText = "bar.warp.sync.b32 0x" + std::to_string(mask) + ", " + std::to_string(reconvergence_pc) + ";";
    return ctx;
}

inline StatementContext make_bar_sync(int bar_id = 0) {
    StatementContext ctx;
    ctx.type = S_BAR;
    BarrierInstr instr;
    instr.barId = bar_id;
    ctx.data = instr;
    ctx.instructionText = "bar.sync " + std::to_string(bar_id) + ";";
    return ctx;
}

inline StatementContext make_mov(const std::string& dst, const std::string& src) {
    StatementContext ctx;
    ctx.type = S_MOV;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    ctx.instructionText = "mov.b32 " + dst + ", " + src + ";";
    return ctx;
}

inline StatementContext make_mov_imm(const std::string& dst, int64_t imm) {
    StatementContext ctx;
    ctx.type = S_MOV;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(imm)}});
    ctx.data = instr;
    ctx.instructionText = "mov.b32 " + dst + ", " + std::to_string(imm) + ";";
    return ctx;
}

inline StatementContext make_add(const std::string& dst, const std::string& src1, const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_ADD;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "add.b32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_mul(const std::string& dst, const std::string& src1, const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_MUL;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "mul.lo.s32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_ld_shared(const std::string& dst_reg, const std::string& shared_var, const std::string& offset_reg) {
    StatementContext ctx;
    ctx.type = S_LD;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, Qualifier::Q_B32};
    std::string addr = "[" + shared_var + "+" + offset_reg + "]";
    instr.operands.push_back(OperandContext{RegOperand{dst_reg, -1}});
    instr.operands.push_back(OperandContext{VariableOperand{addr}});
    ctx.data = instr;
    ctx.instructionText = "ld.shared.b32 " + dst_reg + ", " + addr + ";";
    return ctx;
}

inline StatementContext make_st_shared(const std::string& shared_var, const std::string& offset_reg, const std::string& src_reg) {
    StatementContext ctx;
    ctx.type = S_ST;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, Qualifier::Q_B32};
    std::string addr = "[" + shared_var + "+" + offset_reg + "]";
    instr.operands.push_back(OperandContext{VariableOperand{addr}});
    instr.operands.push_back(OperandContext{RegOperand{src_reg, -1}});
    ctx.data = instr;
    ctx.instructionText = "st.shared.b32 " + addr + ", " + src_reg + ";";
    return ctx;
}

inline StatementContext make_setp_lt(const std::string& pred, const std::string& src1, const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "setp.lt.u32 " + pred + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_bra(const std::string& target) {
    StatementContext ctx;
    ctx.type = S_BRA;
    BranchInstr instr;
    instr.target = target;
    ctx.data = instr;
    ctx.instructionText = "bra " + target + ";";
    return ctx;
}

inline StatementContext make_bra_pred(const std::string& target, const std::string& pred, bool neg = false) {
    StatementContext ctx;
    ctx.type = S_BRA;
    BranchInstr instr;
    instr.target = target;
    instr.predicate = pred;
    instr.predicate_negated = neg;
    ctx.data = instr;
    ctx.instructionText = (neg ? "@!" : "@") + pred + " bra " + target + ";";
    return ctx;
}

inline StatementContext make_label(const std::string& name) {
    StatementContext ctx;
    ctx.type = S_LABEL;
    ctx.data = LabelInstr{name};
    ctx.instructionText = name + ":;";
    return ctx;
}

inline StatementContext make_nop() {
    StatementContext ctx;
    ctx.type = S_PRAGMA;
    ctx.data = PragmaInstr{"nop"};
    ctx.instructionText = "nop;";
    return ctx;
}

inline StatementContext make_exit() {
    StatementContext ctx;
    ctx.type = S_EXIT;
    ctx.data = VoidInstr{};
    ctx.instructionText = "exit;";
    return ctx;
}

// ============================================================================
// Warp Setup Helpers
// ============================================================================

inline void setup_warp(WarpContext& warp,
                      std::vector<std::unique_ptr<ThreadContext>>& threads,
                      int num_lanes = 32,
                      CTAContext* cta = nullptr) {
    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {(uint32_t)num_lanes, 1, 1};

    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;
    std::vector<StatementContext> stmts;

    threads.clear();

    for (int i = 0; i < num_lanes; i++) {
        auto t = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)i, 0, 0};
        t->init(blockIdx, tid, gridDim, blockDim, stmts,
                &name2Sym, label2pc, nullptr, cta);
        t->set_state(RUN);
        threads.push_back(std::move(t));
    }

    for (int i = 0; i < num_lanes; i++) {
        warp.add_thread(std::move(threads[i]), i);
    }

    warp.set_active_mask(0xFFFFFFFF);
}

inline void reset_warp(WarpContext& warp, int num_lanes = 32) {
    for (int i = 0; i < num_lanes; i++) {
        auto* t = warp.get_thread(i);
        if (!t) continue;
        t->pc = 0;
        t->state = RUN;
        warp.get_warp_state().threads[i].pc = 0;
        warp.get_warp_state().threads[i].next_pc = 0;
        warp.get_warp_state().threads[i].is_blocked = false;
        warp.get_warp_state().threads[i].is_active = true;
        warp.get_warp_state().threads[i].is_exited = false;
        warp.get_warp_state().threads[i].status = ThreadStatus::Active;
    }
    warp.set_active_mask(0xFFFFFFFF);
}

// ============================================================================
// Verification Helpers
// ============================================================================

inline int count_active_lanes(const WarpContext& warp) {
    int n = 0;
    for (int i = 0; i < 32; i++)
        if (warp.is_lane_active(i)) n++;
    return n;
}

inline int count_at_pc(const WarpContext& warp, uint32_t pc) {
    int n = 0;
    for (int i = 0; i < 32; i++)
        if (warp.get_warp_state().threads[i].pc == pc) n++;
    return n;
}

inline uint32_t get_active_mask(const WarpContext& warp) {
    return warp.get_active_mask();
}

inline void check_mask(WarpContext& warp, uint32_t expected, const char* msg = nullptr) {
    uint32_t actual = warp.get_active_mask();
    std::ostringstream ss;
    ss << "mask: expected=0x" << std::hex << expected << ", got=0x" << actual;
    if (msg) ss << " (" << msg << ")";
    INFO(ss.str());
    CHECK(actual == expected);
}

// ============================================================================
// Shared Memory Helpers
// ============================================================================

inline void write_shared(void* base, size_t offset, uint32_t val) {
    static_cast<uint32_t*>(base)[offset] = val;
}

inline uint32_t read_shared(void* base, size_t offset) {
    return static_cast<uint32_t*>(base)[offset];
}

inline void* allocate_shared(size_t elems) {
    void* p = malloc(elems * sizeof(uint32_t));
    memset(p, 0, elems * sizeof(uint32_t));
    return p;
}

// ============================================================================
// PTX File Loading Helpers (Mode 2)
// ============================================================================

inline std::string load_ptx_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline std::string extract_ptx_cuobjdump(const std::string& bin_path) {
    std::ostringstream cmd;
    cmd << "cuobjdump -ptx -all " << bin_path << " 2>&1";
    FILE* p = popen(cmd.str().c_str(), "r");
    if (!p) return "";
    char buf[4096];
    std::string r;
    while (fgets(buf, sizeof(buf), p)) r += buf;
    pclose(p);
    return r;
}

inline bool ptx_contains(const std::string& ptx, const std::string& token) {
    return ptx.find(token) != std::string::npos;
}

// ============================================================================
// Statement Sequence Printing (debugging)
// ============================================================================

inline void print_stmts(const std::vector<StatementContext>& stmts, const char* label = "") {
    std::ostringstream ss;
    ss << "=== Statements " << (label ? label : "") << " (count=" << stmts.size() << ") ===" << std::endl;
    for (size_t i = 0; i < stmts.size() && i < 30; i++) {
        ss << "  [" << std::setw(3) << i << "] " << stmts[i].instructionText << std::endl;
    }
    if (stmts.size() > 30)
        ss << "  ... (" << (stmts.size() - 30) << " more)" << std::endl;
    INFO(ss.str());
}

// ============================================================================
// StatementContext 序列执行 (Mode 3 核心)
// ============================================================================

// 需要 g_gpu_context（由 libcudart.so 初始化）
#ifndef GPU_CONTEXT_H
#include "ptxsim/gpu_context.h"
#endif
extern std::unique_ptr<GPUContext> g_gpu_context;

// 从 StatementContext 向量创建 KernelLaunchRequest
inline KernelLaunchRequest make_kernel_request(
    std::vector<StatementContext>& statements,
    std::map<std::string, Symtable*>& name2Sym,
    std::map<std::string, int>& label2pc,
    void** args = nullptr,
    Dim3 gridDim = {1, 1, 1},
    Dim3 blockDim = {32, 1, 1},
    size_t sharedMem = 0) {

  KernelLaunchRequest req;
  req.args = args;
  req.gridDim = gridDim;
  req.blockDim = blockDim;
  req.statements = &statements;
  req.name2Sym = std::make_shared<std::map<std::string, Symtable*>>(name2Sym);
  req.label2pc = std::make_shared<std::map<std::string, int>>(label2pc);
  req.shared_mem_size = sharedMem;
  return req;
}

// 通过 GPUContext 执行一组 StatementContext 指令序列
// 返回：true 表示成功提交并执行完成
inline bool run_statement_sequence(
    std::vector<StatementContext>& statements,
    void** args = nullptr,
    Dim3 gridDim = {1, 1, 1},
    Dim3 blockDim = {32, 1, 1},
    size_t sharedMem = 0) {

  if (!g_gpu_context) return false;

  std::map<std::string, Symtable*> name2Sym;
  std::map<std::string, int> label2pc;

  // 从 statements 提取 labels
  for (size_t i = 0; i < statements.size(); i++) {
    if (statements[i].type == S_LABEL) {
      const auto& lbl = std::get<LabelInstr>(statements[i].data);
      label2pc[lbl.labelName] = static_cast<int>(i);
    }
  }

  auto req = make_kernel_request(statements, name2Sym, label2pc, args, gridDim, blockDim, sharedMem);
  g_gpu_context->submit_kernel_request(std::move(req));
  g_gpu_context->wait_for_completion();
  return true;
}

#endif // TEST_HELPERS_HPP
