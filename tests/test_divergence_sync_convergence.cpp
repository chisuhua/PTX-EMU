/**
 * 指令序列集成测试：验证 divergence_sync kernel 中分歧汇聚行为。
 *
 * 核心原则：
 *   1. 所有 PC 变化通过 execute_warp_instruction → 指令执行管道驱动
 *   2. 路径由 step_warp 完全模拟调度器决策（sm_context.cpp:250-264）
 *   3. 测试不干预调度器选择，只验证其选择是否正确
 *   4. predicate 通过 RegisterBankManager 设置 per-lane 值
 *   5. 分歧由 handle_branch 自动处理
 *
 * 指令布局（35 条）：
 *   PC=0..3:    MOV（分歧前）
 *   PC=4:       @%p1 bra $L__BB0_4（分歧）→ taken=PC=28, not_taken=PC=5
 *   PC=5..13:   MOV（Path A）
 *   PC=14..26:  MOV（汇聚后代码）
 *   PC=27:      ret
 *   PC=28..33:  MOV（Path B）
 *   PC=34:      bra.uni $L__BB0_3（→PC=14）
 */
#include "catch_amalgamated.hpp"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/simt_stack.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_factory.h"
#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <cstdint>

static constexpr int      BRANCH_PC     = 4;
static constexpr int      CONV_PC       = 14;
static constexpr int      PATH_A_START  = 5;
static constexpr int      PATH_A_END    = 13;
static constexpr int      PATH_B_TARGET = 28;
static constexpr int      PATH_B_END    = 33;
static constexpr int      BRA_UNI_PC    = 34;
static constexpr int      NUM_STMTS     = 35;

// ============================================================================
// 辅助
// ============================================================================
static void init_factory() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        ptxsim::DebugConfig::get().set_trace_simt_stack_enabled(true);
        ptxsim::DebugConfig::get().set_trace_divergence_enabled(true);
        ptxsim::LoggerConfig::get().set_component_level("emu", ptxsim::log_level::debug);
        done = true;
    }
}

static StatementContext nop() {
    using namespace ptxir::factory;
    return makeGenericInstr(
        S_MOV,
        {Qualifier::Q_U32},
        {OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"0"}}},
        "mov.u32 %r1, 0;");
}

static std::vector<StatementContext> build_instrs(
    std::map<std::string, int> &l2pc)
{
    std::vector<StatementContext> v;
    v.reserve(NUM_STMTS);
    for (int i = 0; i < NUM_STMTS; i++) v.push_back(nop());
    // PC=4: @%p1 bra $L__BB0_4
    {
        auto b = ptxir::factory::makeBranchInstr(
            S_BRA, {}, "L__BB0_4", "%p1", false, "@%p1 bra $L__BB0_4;");
        std::get<BranchInstr>(b.data).reconvergence_pc = CONV_PC;
        v[BRANCH_PC] = std::move(b);
    }
    // PC=34: bra.uni $L__BB0_3
    {
        auto b = ptxir::factory::makeBranchInstr(
            S_BRA, {}, "L__BB0_3", "", false, "bra.uni $L__BB0_3;");
        v[BRA_UNI_PC] = std::move(b);
    }
    // PC=27: ret
    {
        StatementContext r;
        r.type = S_RET; r.data = VoidInstr{};
        r.instructionText = "ret;";
        v[27] = std::move(r);
    }
    l2pc["L__BB0_4"] = PATH_B_TARGET;
    l2pc["L__BB0_3"] = CONV_PC;
    return v;
}

static void setup_pred(WarpContext *w, uint32_t taken) {
    auto rbm = w->get_register_bank_manager();
    REQUIRE(rbm);
    rbm->create_register("p1", 1);
    for (int i = 0; i < 32; i++) {
        auto *p = static_cast<uint8_t*>(rbm->get_register("p1", 0, i));
        REQUIRE(p); *p = (taken & (1u << i)) ? 1 : 0;
    }
}

static WarpContext* setup(SMContext &sm,
                          std::vector<StatementContext> &v,
                          std::map<std::string, int> &l2pc)
{
    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1,1,1}, b{32,1,1}, bi{0,0,0};
    std::map<std::string, Symtable*> n2s;
    blk->init(g, b, bi, v, &n2s, l2pc);
    blk->sharedMemBytes = 1024;
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok); return sm.get_warp(0);
}

// step_warp — 完全模拟 sm_context.cpp 调度器算法
static int step_warp(WarpContext *w, std::vector<StatementContext> &v) {
    auto m = w->get_lanes_by_pc();
    REQUIRE_FALSE(m.empty());
    int pick = m.begin()->first;
    auto &ws = w->get_warp_state();
    for (auto &[pc, lanes] : m) {
        bool ok = true;
        for (int l : lanes) { if (ws.threads[l].is_blocked) { ok = false; break; } }
        if (ok) { pick = pc; break; }
    }
    w->execute_warp_instruction(v[pick], pick);
    while (w->check_reconvergence()) {}
    return pick;
}

// ============================================================================
// Test A: 分歧 → Path A 到汇聚点阻塞 → 调度器切 Path B → Path B 到达汇聚
// ============================================================================
TEST_CASE("scheduler switches at convergence point",
          "[divergence][convergence][integrated][scheduler]")
{
    init_factory(); ResourceManager::instance().initialize(1, 8192);
    std::map<std::string, int> l2pc;
    auto v = build_instrs(l2pc);
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup(sm, v, l2pc);
    setup_pred(w, 0x0000FFFFu);

    // 分歧前: step_warp 从 warp 状态获取 PC
    CHECK(step_warp(w, v) == 0);
    CHECK(step_warp(w, v) == 1);
    CHECK(step_warp(w, v) == 2);
    CHECK(step_warp(w, v) == 3);
    CHECK(step_warp(w, v) == BRANCH_PC);  // @%p1 bra → 分歧
    REQUIRE(w->get_simt_stack().depth() == 1);

    // 分歧后: Path A(16-31)→PC=5, Path B(0-15)→PC=28
    CHECK(w->get_thread_pc(31) == PATH_A_START);
    CHECK(w->get_thread_pc(0)  == PATH_B_TARGET);
    { auto pm = w->get_lanes_by_pc(); CHECK(pm.size() == 2); }

    // step_warp 自然选 Path A（PC=5 最低）→ 执行 PC=5..13
    CHECK(step_warp(w, v) == PATH_A_START);
    CHECK(step_warp(w, v) == PATH_A_START + 1);
    CHECK(step_warp(w, v) == PATH_A_START + 2);
    CHECK(step_warp(w, v) == PATH_A_START + 3);
    CHECK(step_warp(w, v) == PATH_A_START + 4);
    CHECK(step_warp(w, v) == PATH_A_START + 5);
    CHECK(step_warp(w, v) == PATH_A_START + 6);
    CHECK(step_warp(w, v) == PATH_A_START + 7);
    CHECK(step_warp(w, v) == PATH_A_END);

    // Path A 到达 PC=14（汇聚点），Path B 仍在 PC=28
    CHECK(w->get_thread_pc(16) == CONV_PC);
    CHECK(w->get_thread_pc(0)  == PATH_B_TARGET);
    CHECK(w->check_reconvergence() == false);

    // === 汇聚点阻塞 + 调度器切换 ===
    // step_warp 选 PC=14（最低），check_and_block_at_reconvergence_point
    // 阻塞 Path A（早到汇聚点，等待 Path B）
    int pc = step_warp(w, v);
    CHECK(pc == CONV_PC);
    // Path A lanes (16-31) 被阻塞在 PC=14
    CHECK(w->get_warp_state().threads[16].is_blocked == true);
    CHECK(w->get_warp_state().threads[20].is_blocked == true);
    // get_lanes_by_pc 不再包含 Path A（被阻塞）
    { auto pm = w->get_lanes_by_pc();
      CHECK(pm.size() == 1);          // 只有 Path B
      CHECK(pm.count(PATH_B_TARGET) == 1); }

    // === 调度器切至 Path B（PC=28）===
    CHECK(step_warp(w, v) == PATH_B_TARGET);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 1);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 2);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 3);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 4);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 5);
    // bra.uni → 跳转到 PC=14
    CHECK(step_warp(w, v) == BRA_UNI_PC);

    // === 汇聚 ===
    // Path B 到达 PC=14 → is_converged true → entry 弹出
    // check_reconvergence 解阻 Path A（lines 114-119）
    CHECK(w->get_simt_stack().empty());
    CHECK(w->get_exec_mask() == 0xFFFFFFFFu);
    // Path A 已解阻
    CHECK(w->get_warp_state().threads[16].is_blocked == false);
    CHECK(w->get_warp_state().threads[16].is_active == true);
    // 所有 32 线程在 PC=14
    for (int i = 0; i < 32; i++) CHECK(w->get_thread_pc(i) == CONV_PC);
    { auto pm = w->get_lanes_by_pc(); CHECK(pm.size() == 1); CHECK(pm.count(CONV_PC) == 1); }
}

// ============================================================================
// Test B: 两级分歧 — loop entry + primary entry，两级汇聚点阻塞 + 调度器切换
// ============================================================================
TEST_CASE("two level div with convergence block",
          "[divergence][convergence][integrated]")
{
    init_factory(); ResourceManager::instance().initialize(1, 8192);
    std::map<std::string, int> l2pc;
    auto v = build_instrs(l2pc);
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup(sm, v, l2pc);
    setup_pred(w, 0x0000FFFFu);

    step_warp(w, v); step_warp(w, v);
    step_warp(w, v); step_warp(w, v);
    step_warp(w, v);  // @%p1 bra → 分歧

    // 模拟循环 back-edge 推入的二级分歧 entry
    ptxsim::SIMTStackEntry le;
    le.branch_pc = 13; le.reconvergence_pc = CONV_PC;
    le.active_mask = 0xFFFC0000u; // lanes 18-31
    le.return_mask = 0xFFFF0000u; le.return_pc = CONV_PC;
    w->get_simt_stack().push(le);
    REQUIRE(w->get_simt_stack().depth() == 2);

    // step_warp 自然选 Path A（PC=5 最低）
    int pc;
    pc = step_warp(w, v); CHECK(pc == PATH_A_START);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 1);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 2);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 3);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 4);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 5);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 6);
    pc = step_warp(w, v); CHECK(pc == PATH_A_START + 7);
    pc = step_warp(w, v); CHECK(pc == PATH_A_END);

    // Path A 到 PC=14 → loop entry 的 active_mask (18-31) 全到 → 汇聚弹出
    CHECK(w->get_simt_stack().depth() == 1);

    // 此时栈顶为 primary entry (active_mask=0x0000FFFF)
    // Path A 在 PC=14 → check_and_block 阻塞 Path A
    pc = step_warp(w, v);
    CHECK(pc == CONV_PC);
    CHECK(w->get_warp_state().threads[16].is_blocked == true);

    // 调度器切至 Path B
    CHECK(step_warp(w, v) == PATH_B_TARGET);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 1);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 2);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 3);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 4);
    CHECK(step_warp(w, v) == PATH_B_TARGET + 5);
    CHECK(step_warp(w, v) == BRA_UNI_PC);

    // 汇聚：primary entry 弹出，Path A 解阻
    CHECK(w->get_simt_stack().empty());
    CHECK(w->get_exec_mask() == 0xFFFFFFFFu);
    CHECK(w->get_warp_state().threads[16].is_blocked == false);
}

// ============================================================================
// Test C: 验证调度器分歧后必须选 Path A（最低 PC 非阻塞分组）
// ============================================================================
TEST_CASE("scheduler picks lowest non-blocked PC group after divergence",
          "[divergence][convergence][integrated][scheduler]")
{
    init_factory(); ResourceManager::instance().initialize(1, 8192);
    std::map<std::string, int> l2pc;
    auto v = build_instrs(l2pc);
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup(sm, v, l2pc);
    setup_pred(w, 0x0000FFFFu);

    step_warp(w, v); step_warp(w, v);
    step_warp(w, v); step_warp(w, v);
    CHECK(step_warp(w, v) == BRANCH_PC);  // 分歧

    // 分歧后 get_lanes_by_pc = {5: [16..31], 28: [0..15]}
    // 调度器必须选 PC=5（最低非阻塞分组），否则调度器有 bug
    CHECK(step_warp(w, v) == PATH_A_START);
    CHECK(step_warp(w, v) == PATH_A_START + 1);
    CHECK(step_warp(w, v) == PATH_A_START + 2);
    CHECK(step_warp(w, v) == PATH_A_START + 3);
    CHECK(step_warp(w, v) == PATH_A_START + 4);
    CHECK(step_warp(w, v) == PATH_A_START + 5);
    CHECK(step_warp(w, v) == PATH_A_START + 6);
    CHECK(step_warp(w, v) == PATH_A_START + 7);
    CHECK(step_warp(w, v) == PATH_A_END);

    // Path A 到 PC=14 → 阻塞
    CHECK(step_warp(w, v) == CONV_PC);
    CHECK(w->get_warp_state().threads[16].is_blocked == true);

    // 调度器切 Path B
    for (int i = 0; i < 6; i++) step_warp(w, v);
    CHECK(step_warp(w, v) == BRA_UNI_PC);
    CHECK(w->get_simt_stack().empty());
}

// ============================================================================
// Test D: 边界 — active_mask 全部到达才收敛
// ============================================================================
TEST_CASE("boundary conv requires all active mask",
          "[divergence][convergence][integrated][edge]")
{
    init_factory(); ResourceManager::instance().initialize(1, 8192);
    std::map<std::string, int> l2pc;
    auto v = build_instrs(l2pc);
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup(sm, v, l2pc);
    setup_pred(w, 0x0000FFFFu);

    step_warp(w, v); step_warp(w, v);
    step_warp(w, v); step_warp(w, v);
    step_warp(w, v); // @%p1 bra → 分歧

    // Path A → PC=14 → 阻塞
    for (int i = 0; i < 9; i++) step_warp(w, v);
    step_warp(w, v); // 阻塞

    // Path B → bra.uni → 汇聚
    for (int i = 0; i < 6; i++) step_warp(w, v);
    CHECK(step_warp(w, v) == BRA_UNI_PC);
    CHECK(w->get_simt_stack().empty());
}

// ============================================================================
// Test E: 边界 — active_mask 之外不影响收敛判定（not_taken 先到不影响）
// ============================================================================
TEST_CASE("boundary non active mask no conv effect",
          "[divergence][convergence][integrated][edge]")
{
    init_factory(); ResourceManager::instance().initialize(1, 8192);
    std::map<std::string, int> l2pc;
    auto v = build_instrs(l2pc);
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup(sm, v, l2pc);
    setup_pred(w, 0x0000FFFFu);

    step_warp(w, v); step_warp(w, v);
    step_warp(w, v); step_warp(w, v);
    step_warp(w, v); // 分歧

    // Path A 到 PC=14（不触发收敛，因为 active_mask 跟踪 taken=0-15）
    for (int i = 0; i < 9; i++) step_warp(w, v);
    CHECK(w->check_reconvergence() == false);

    // Path A 被阻塞 → 调度器切 Path B
    step_warp(w, v); // 阻塞
    for (int i = 0; i < 6; i++) step_warp(w, v);
    CHECK(step_warp(w, v) == BRA_UNI_PC);
    CHECK(w->get_simt_stack().empty());
}
