/**
 * @file test_divergence_sync_standalone_integrated.cpp
 * @brief 对标 test_divergence_sync_standalone.cu 的集成测试
 *
 * 复现 kernel 的分歧→屏障→重汇聚模式:
 * - Lane 0-15:   计算 sum 0..lane
 * - Lane 16-31:  计算 product 1..(lane-15)
 * - bar.warp.sync: warp 级屏障同步
 * - Thread 0:    归约所有 32 个结果
 *
 * ADR-0013 合规: 所有 StatementContext 通过 ptxir::factory 构造，
 * 不得自行实现本地 make_barrier_stmt 等重复模式。
 *
 * 测试策略: execute_warp_instruction 要求所有线程在同一 PC。
 * 分歧分支通过 SIMT stack 推送处理，barrier 执行前手动汇聚线程到 barrier PC。
 */

// NOTE: 原 test_divergence_sync_isolated.cpp 的屏障同步测试已合并到此处。
// 该文件使用 setp+selp 谓词化（非真实 SIMT 分歧），其屏障行为覆盖已由本文件的
// "barrier releases all threads" 和 "full warp barrier-then-divergence flow" 覆盖。

#include "catch_amalgamated.hpp"
#include "ptx_ir/statement_factory.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/predicates.h"
#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"
#include <array>
#include <map>
#include <memory>
#include <vector>
#include <cstdint>

using ptxsim::testing::step_warp;
using ptxsim::testing::setup_pred;

namespace {
using namespace ptxir::factory;

// ============================================================================
// 初始化
// ============================================================================

static void init_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

// ============================================================================
// 构建对标 kernel 的指令序列
//
// Kernel PTX 核心结构:
//   mov %r_tid, %tid.x
//   setp.lt %p_lt16, %r_tid, 16
//   @%p_lt16  bra L_path_a
//   bra L_path_b
// L_path_a:
//   mov %r_val, %r_tid
//   bra L_join
// L_path_b:
//   mov %r_val, 1
//   bra L_join
// L_join:
//   st.shared [%r_tid], %r_val
//   bar.warp.sync
//   setp.eq %p_t0, %r_tid, 0
//   @%p_t0 bra L_reduce
//   bra L_exit
// L_reduce:
//   ld.shared %r_tmp, [%r_tid]
//   add %r_sum, %r_sum, %r_tmp
//   bra L_exit
// L_exit:
//   ret
// ============================================================================

static std::vector<StatementContext> build_divergence_sync_statements() {
    std::vector<StatementContext> stmts;

    // PC=0: mov %r_tid, %tid.x
    stmts.push_back(makeGenericInstr(S_MOV,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 1}}, OperandContext{RegOperand{"tid.x", -1}}},
        "mov.u32 %r_tid, %tid.x;"));

    // PC=1: setp.lt %p_lt16, %r_tid, 16
    stmts.push_back(makeGenericInstr(S_SETP,
        {Qualifier::Q_B32, Qualifier::Q_LT},
        {OperandContext{RegOperand{"p_lt16"}},
         OperandContext{RegOperand{"r", 1}},
         OperandContext{ImmOperand{"16"}}},
        "setp.lt.u32 %p_lt16, %r_tid, 16;"));

    // PC=2: @%p_lt16 bra L_path_a
    stmts.push_back(makeBranchInstr(S_BRA,
        {},
        "L_path_a",
        "%p_lt16",
        false,
        "@%p_lt16 bra L_path_a;"));

    // PC=3: bra L_path_b
    stmts.push_back(makeBranchInstr(S_BRA, {}, "L_path_b", "", false, "bra L_path_b;"));

    // PC=4: L_path_a:
    stmts.push_back(makeLabelInstr("L_path_a", "L_path_a:"));

    // PC=5: mov %r_val, %r_tid
    stmts.push_back(makeGenericInstr(S_MOV,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 2}}, OperandContext{RegOperand{"r", 1}}},
        "mov.u32 %r_val, %r_tid;"));

    // PC=6: bra L_join
    stmts.push_back(makeBranchInstr(S_BRA, {}, "L_join", "", false, "bra L_join;"));

    // PC=7: L_path_b:
    stmts.push_back(makeLabelInstr("L_path_b", "L_path_b:"));

    // PC=8: mov %r_val, 1
    stmts.push_back(makeGenericInstr(S_MOV,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 2}}, OperandContext{ImmOperand{"1"}}},
        "mov.u32 %r_val, 1;"));

    // PC=9: bra L_join
    stmts.push_back(makeBranchInstr(S_BRA, {}, "L_join", "", false, "bra L_join;"));

    // PC=10: L_join:
    stmts.push_back(makeLabelInstr("L_join", "L_join:"));

    // PC=11: st.shared [%r_tid], %r_val
    stmts.push_back(makeGenericInstr(S_ST,
        {Qualifier::Q_SHARED, Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 2}},
         OperandContext{RegOperand{"r", 1}}},
        "st.shared.u32 [%r_tid], %r_val;"));

    // PC=12: bar.warp.sync (reconvergence at PC=13)
    stmts.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 13));

    // PC=13: setp.eq %p_t0, %r_tid, 0
    stmts.push_back(makeGenericInstr(S_SETP,
        {Qualifier::Q_B32, Qualifier::Q_EQ},
        {OperandContext{RegOperand{"p_t0"}},
         OperandContext{RegOperand{"r", 1}},
         OperandContext{ImmOperand{"0"}}},
        "setp.eq.u32 %p_t0, %r_tid, 0;"));

    // PC=14: @%p_t0 bra L_reduce
    stmts.push_back(makeBranchInstr(S_BRA,
        {},
        "L_reduce",
        "%p_t0",
        false,
        "@%p_t0 bra L_reduce;"));

    // PC=15: bra L_exit
    stmts.push_back(makeBranchInstr(S_BRA, {}, "L_exit", "", false, "bra L_exit;"));

    // PC=16: L_reduce:
    stmts.push_back(makeLabelInstr("L_reduce", "L_reduce:"));

    // PC=17: ld.shared %r_tmp, [%r_tid]
    stmts.push_back(makeGenericInstr(S_LD,
        {Qualifier::Q_SHARED, Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 3}},
         OperandContext{RegOperand{"r", 1}}},
        "ld.shared.u32 %r_tmp, [%r_tid];"));

    // PC=18: add %r_sum, %r_sum, %r_tmp
    stmts.push_back(makeGenericInstr(S_ADD,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 4}},
         OperandContext{RegOperand{"r", 4}},
         OperandContext{RegOperand{"r", 3}}},
        "add.u32 %r_sum, %r_sum, %r_tmp;"));

    // PC=19: bra L_exit
    stmts.push_back(makeBranchInstr(S_BRA, {}, "L_exit", "", false, "bra L_exit;"));

    // PC=20: L_exit:
    stmts.push_back(makeLabelInstr("L_exit", "L_exit:"));

    // PC=21: ret
    stmts.push_back(makeVoidInstr(S_RET, "ret;"));

    return stmts;
}

// ============================================================================
// Warp 上下文辅助
// ============================================================================

static WarpContext* create_warp_with_threads(
    SMContext& sm, std::unique_ptr<CTAContext> block,
    std::shared_ptr<RegisterBankManager> register_bank) {
    block->sharedMemBytes = 128;
    bool ok = sm.add_block(std::move(block));
    REQUIRE(ok == true);
    WarpContext* warp = sm.get_warp(0);
    warp->set_register_bank_manager(register_bank);
    for (int i = 0; i < 32; i++) {
        warp->get_thread(i)->set_register_bank_manager(register_bank);
    }
    return warp;
}

// 从语句列表中构建 label → PC 的映射表
static std::map<std::string, int> build_label2pc(
    const std::vector<StatementContext>& stmts) {
    std::map<std::string, int> map;
    for (int i = 0; i < static_cast<int>(stmts.size()); i++) {
        if (stmts[i].type == S_LABEL) {
            const LabelInstr& li = std::get<LabelInstr>(stmts[i].data);
            map[li.labelName] = i;
        }
    }
    return map;
}

static std::unique_ptr<CTAContext> create_block(
    std::vector<StatementContext>& statements,
    Dim3 gridDim = {1, 1, 1},
    Dim3 blockDim = {32, 1, 1},
    Dim3 blockIdx = {0, 0, 0}) {
    auto block = std::make_unique<CTAContext>();
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc = build_label2pc(statements);
    block->init(gridDim, blockDim, blockIdx, statements, &name2Sym, label2pc);
    return block;
}

// 将所有线程推进到指定 PC（模拟 SIMT stack 汇聚后的调度行为）
static void advance_all_to_pc(WarpContext& warp, int pc) {
    for (int i = 0; i < 32; i++) {
        warp.get_warp_state().threads[i].pc = pc;
        warp.get_warp_state().threads[i].next_pc = pc;
    }
}

} // anonymous namespace

// ============================================================================
// 测试用例
// ============================================================================

TEST_CASE("divergence_sync_standalone: barrier releases all threads to reconvergence PC", "[divergence_sync][barrier]") {
    init_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    auto statements = build_divergence_sync_statements();

    auto register_bank = std::make_shared<RegisterBankManager>(1, 32);
    auto registers = RegisterAnalyzer::analyze_registers(statements);
    register_bank->preallocate_registers(registers);

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements), register_bank);

    warp->set_active_mask(0xFFFFFFFF);

    // 将所有线程汇聚到 barrier PC=12
    advance_all_to_pc(*warp, 12);

    // 执行 bar.warp.sync: 所有 32 线程同时到达，barrier 立即完成（直接控制 PC）
    warp->execute_warp_instruction(statements[12], 12);

    // 验证: barrier 后所有线程在 PC=13
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 13);
    }
}

TEST_CASE("divergence_sync_standalone: thread 0-only branch after barrier", "[divergence_sync][barrier][predication]") {
    init_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    auto statements = build_divergence_sync_statements();

    auto register_bank = std::make_shared<RegisterBankManager>(1, 32);
    auto registers = RegisterAnalyzer::analyze_registers(statements);
    register_bank->preallocate_registers(registers);

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements), register_bank);

    warp->set_active_mask(0xFFFFFFFF);

    // 手动初始化 r1 (tid) 寄存器，因为测试从 PC=12 开始，跳过了前面的 mov 指令
    for (int i = 0; i < 32; i++) {
        void* tid_addr = register_bank->get_register("r1", 0, i);
        if (tid_addr) {
            *static_cast<uint32_t*>(tid_addr) = i;
        }
    }

    // 汇聚到 barrier，执行 barrier（直接控制 PC）
    advance_all_to_pc(*warp, 12);
    warp->execute_warp_instruction(statements[12], 12);
    REQUIRE(warp->get_thread(0)->get_pc() == 13);

    // 执行 PC=13: setp.eq %p_t0, %r_tid, 0（直接控制 PC）
    warp->execute_warp_instruction(statements[13], 13);

    // 执行 PC=14: @%p_t0 bra L_reduce（直接控制 PC）
    warp->execute_warp_instruction(statements[14], 14);

    // 验证: thread 0 (tid=0) 在 PC=16 (L_reduce)，其他线程在 PC=15 (bra L_exit)
    CHECK(warp->get_thread(0)->get_pc() == 16);
    CHECK(warp->get_thread(1)->get_pc() == 15);
    CHECK(warp->get_thread(16)->get_pc() == 15);
    CHECK(warp->get_thread(31)->get_pc() == 15);
}

TEST_CASE("divergence_sync_standalone: full warp barrier-then-divergence flow", "[divergence_sync][execution][full]") {
    init_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    auto statements = build_divergence_sync_statements();

    auto register_bank = std::make_shared<RegisterBankManager>(1, 32);
    auto registers = RegisterAnalyzer::analyze_registers(statements);
    register_bank->preallocate_registers(registers);

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements), register_bank);

    warp->set_active_mask(0xFFFFFFFF);

    // 手动初始化 r1 (tid) 寄存器，因为测试从 PC=11 开始，跳过了前面的 mov 指令
    for (int i = 0; i < 32; i++) {
        void* tid_addr = register_bank->get_register("r1", 0, i);
        if (tid_addr) {
            *static_cast<uint32_t*>(tid_addr) = i;
        }
    }

    // Phase 1: 执行 barrier 前的 store (PC=11)（直接控制 PC）
    advance_all_to_pc(*warp, 11);
    warp->execute_warp_instruction(statements[11], 11);

    // Phase 2: barrier (PC=12) - 所有线程同步（直接控制 PC）
    warp->execute_warp_instruction(statements[12], 12);
    REQUIRE(warp->get_thread(0)->get_pc() == 13);

    // Phase 3: barrier 后的分歧 - thread 0 vs 其他线程（直接控制 PC）
    warp->execute_warp_instruction(statements[13], 13); // setp.eq
    warp->execute_warp_instruction(statements[14], 14); // @%p_t0 bra L_reduce

    INFO("After predicated bra (PC=14):");
    for (int lane = 0; lane < 32; lane++) {
        INFO("  lane " << lane << ": pc=" << warp->get_thread(lane)->get_pc());
    }

    // Thread 0 在 L_reduce (PC=16)，其他在 L_exit (PC=15)
    CHECK(warp->get_thread(0)->get_pc() == 16);
    CHECK(warp->get_thread(1)->get_pc() == 15);
    CHECK(warp->get_thread(31)->get_pc() == 15);

    // Phase 4: Thread 0 执行 reduction (PC=16~21)（直接控制 PC）
    warp->execute_warp_instruction(statements[16], 16); // L_reduce: ld.shared
    warp->execute_warp_instruction(statements[17], 17); // add
    warp->execute_warp_instruction(statements[18], 18); // bra L_exit
    warp->execute_warp_instruction(statements[19], 19); // bra L_exit
    warp->execute_warp_instruction(statements[20], 20); // L_exit label
    warp->execute_warp_instruction(statements[21], 21); // ret

    CHECK(warp->get_thread(0)->get_pc() == 22);
    // 其他线程仍在 L_exit (bra L_exit)
    CHECK(warp->get_thread(1)->get_pc() == 15);
}

TEST_CASE("divergence_sync_standalone: all three modes produce same reconvergence", "[divergence_sync][all_modes][reconvergence]") {
    init_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    // 指令布局（与 test_divergence_sync_convergence.cpp 对齐）：
    //   PC=0..3:    4 NOPs（分歧前）
    //   PC=4:       @%p1 bra L_reduce（reconvergence_pc=14）→ SIMT 分歧
    //   PC=5..13:   9 NOPs（Path A，predicate 不取 lanes 16-31）
    //   PC=14:      NOP（CONV_PC 汇聚点）
    //   PC=28..33:  6 NOPs（Path B，predicate 取 lanes 0-15）
    //   PC=34:      bra.uni L_join → 跳回 CONV_PC=14 触发汇聚
    static constexpr int BRANCH_PC     = 4;
    static constexpr int CONV_PC       = 14;
    static constexpr int PATH_A_START  = 5;
    static constexpr int PATH_A_END    = 13;
    static constexpr int PATH_B_TARGET = 28;
    static constexpr int PATH_B_END    = 33;
    static constexpr int BRA_UNI_PC    = 34;
    static constexpr int NUM_STMTS     = 35;

    std::map<std::string, int> l2pc;
    std::vector<StatementContext> stmts;
    stmts.reserve(NUM_STMTS);
    for (int i = 0; i < NUM_STMTS; i++) {
        stmts.push_back(ptxsim::testing::make_nop());
    }
    stmts[BRANCH_PC] = ptxsim::testing::make_bra_pred("L_reduce", "%p1", false, CONV_PC);
    stmts[BRA_UNI_PC] = ptxsim::testing::make_bra("L_join");
    l2pc["L_reduce"]  = PATH_B_TARGET;
    l2pc["L_join"]    = CONV_PC;

    std::array<ptxsim::DivergenceExecutionMode, 3> modes = {
        ptxsim::DivergenceExecutionMode::Sequential,
        ptxsim::DivergenceExecutionMode::Interleaved,
        ptxsim::DivergenceExecutionMode::ShortestFirst
    };

    for (auto mode : modes) {
        INFO("Mode " << static_cast<int>(mode) << " — divergence → reconvergence");
        SMContext sm(4, 128, 4096, 0);
        sm.set_divergence_execution_mode(mode);
        REQUIRE(sm.get_divergence_execution_mode() == mode);

        auto blk = std::make_unique<CTAContext>();
        Dim3 g{1, 1, 1}, b{32, 1, 1}, bi{0, 0, 0};
        std::map<std::string, Symtable*> n2s;
        blk->init(g, b, bi, stmts, &n2s, l2pc);
        blk->sharedMemBytes = 1024;
        bool ok = sm.add_block(std::move(blk));
        REQUIRE(ok);
        WarpContext* w = sm.get_warp(0);

        setup_pred(w, 0x0000FFFFu);

        CHECK(step_warp(w, stmts) == 0);
        CHECK(step_warp(w, stmts) == 1);
        CHECK(step_warp(w, stmts) == 2);
        CHECK(step_warp(w, stmts) == 3);

        CHECK(step_warp(w, stmts) == BRANCH_PC);
        REQUIRE(w->get_simt_stack().depth() == 1);
        CHECK(w->get_thread_pc(16) == PATH_A_START);
        CHECK(w->get_thread_pc(0)  == PATH_B_TARGET);

        for (int pc = PATH_A_START; pc <= PATH_A_END; pc++) {
            CHECK(step_warp(w, stmts) == pc);
        }
        CHECK(step_warp(w, stmts) == CONV_PC);
        CHECK(w->get_warp_state().threads[16].is_blocked == true);
        CHECK(w->get_warp_state().threads[20].is_blocked == true);
        CHECK(w->get_warp_state().threads[31].is_blocked == true);

        for (int pc = PATH_B_TARGET; pc <= PATH_B_END; pc++) {
            CHECK(step_warp(w, stmts) == pc);
        }
        CHECK(step_warp(w, stmts) == BRA_UNI_PC);

        CHECK(w->get_simt_stack().empty());
        CHECK(w->get_exec_mask() == 0xFFFFFFFFu);
        CHECK(w->get_warp_state().threads[16].is_blocked == false);
        CHECK(w->get_warp_state().threads[16].is_active == true);
        for (int i = 0; i < 32; i++) {
            CHECK(w->get_thread_pc(i) == CONV_PC);
        }

        INFO("Mode " << static_cast<int>(mode) << " completed");
    }
}
