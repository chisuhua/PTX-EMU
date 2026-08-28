// tests/integration/divergence/test_post_barrier_reconvergence_simplegemm.cpp
//
// 指令序列集成测试：simpleGEMM 场景的 divergence + barrier reconciliation
//
// 复现的 bug 模式（与实际 simpleGEMM-int 运行日志匹配）：
//   1. 分支指令按 lane 0-15 vs 16-31 分裂
//   2. Lanes 0-15 进入 a_tile/b_tile 循环（长 divergent 路径）
//   3. Lanes 16-31 跳过循环，直接到 barrier PC
//   4. Lanes 16-31 命中 barrier → 触发 force_reconvergence
//   5. 调度器跳过被阻塞的 PC=barrier，执行 lanes 0-15 的循环
//   6. Lanes 0-15 完成循环到达 barrier
//   7. 此时 wbar 已初始化（buggy: mask=0xFFFF0000），force_reconvergence 路径被跳过
//   8. 所有 lanes 都"arrive" → barrier "完成"
//   9. 但 release 路径仅遍历 arrived_mask & is_active，lanes 0-15 已 inactive → 漏掉
//  10. Lanes 0-15 永远卡在 barrier PC，调度器永不再执行它们
//  11. Lanes 16-31 进入 GEMM 主循环（PC=$L__BB0_8 = 84），但不写入 c
//
// 期望（修复后）：所有 32 个 lane 都被推进到 reconv_pc=70，调度器继续执行
// 它们进入 GEMM 主循环（PC=84）。
//
// 此测试使用 step_warp() 完整驱动调度器 + 屏障处理器，端到端验证
// 32 个 lane 都能到达 PC=84 并执行 ret。
#include "catch_amalgamated.hpp"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/simt_stack.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/predicates.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"
#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <cstdint>

using ptxsim::testing::step_warp;
using ptxsim::testing::setup_pred;
using ptxsim::testing::make_nop;
using ptxsim::testing::make_bra;
using ptxsim::testing::make_bra_pred;
using ptxsim::testing::make_bar_warp_sync;
using ptxsim::testing::make_ret;

namespace {

// 简单镜像 simpleGEMM PC 布局的常量
constexpr int BRANCH_PC     = 1;   // 分支指令
constexpr int PATH_A_START  = 2;   // lanes 0-15 a_tile 循环入口（多个 nop 模拟）
constexpr int PATH_A_END    = 16;  // a_tile 循环出口
constexpr int PATH_B_START  = 17;  // lanes 16-31 路径入口（少量 nop）
constexpr int PATH_B_END    = 23;  // lanes 16-31 路径出口
constexpr int BARRIER_PC    = 24;  // barrier（simpleGEMM 在 PC=69，这里压缩版）
constexpr int RECONV_PC     = 25;  // barrier 后第一个 PC
constexpr int MAIN_LOOP_PC  = 35;  // GEMM 主循环（对应 simpleGEMM 的 PC=84）
constexpr int RET_PC        = 36;
constexpr int NUM_STMTS     = RET_PC + 1;

static void init_factory() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        ResourceManager::instance().initialize(1, 8192);
        done = true;
    }
}

// 构建指令序列：
//   PC=0:        nop（入口）
//   PC=1:        bra_pred PATH_B_START  ← 通过 setup_pred 决定 lanes 16-31 走 PATH_B
//   PC=2..15:    nop * 14（PATH_A，lanes 0-15 走的路径）
//   PC=16:       bra BARRIER_PC         ← lanes 0-15 跳到 barrier
//   PC=17..22:   nop * 6（PATH_B，lanes 16-31 走的路径，比 PATH_A 短）
//   PC=23:       bra BARRIER_PC         ← lanes 16-31 跳到 barrier
//   PC=24:       bar.warp.sync 0xFFFFFFFF, 25   ← 两半汇合
//   PC=25..34:   nop * 10（post-barrier 阶段）
//   PC=35:       nop（GEMM 主循环开始，MAIN_LOOP_PC）
//   PC=36:       ret
//
// 所有 lane 必须最终到达 PC=35 (MAIN_LOOP_PC) 并执行 ret。
static std::vector<ptxemu::ir::StatementContext> build_instrs(
    std::map<std::string, int>& l2pc)
{
    std::vector<ptxemu::ir::StatementContext> v(NUM_STMTS);
    for (auto& s : v) s = make_nop();
    // 分支：lanes 16-31 (p1=true) 跳到 PATH_B_START
    v[BRANCH_PC]   = make_bra_pred("L_PATH_B", "%p1", false, PATH_A_END + 2);
    // PATH_A → bra 到 barrier
    v[PATH_A_END]  = make_bra("L_BARRIER");
    // PATH_B → bra 到 barrier
    v[PATH_B_END]  = make_bra("L_BARRIER");
    // 屏障：所有 32 个 lane 都参与，reconv_pc=25
    v[BARRIER_PC]  = make_bar_warp_sync(0xFFFFFFFFu, RECONV_PC);
    // MAIN_LOOP_PC 已设为 nop（默认），ret 在 RET_PC
    v[RET_PC]      = make_ret();
    l2pc["L_PATH_B"]   = PATH_B_START;
    l2pc["L_BARRIER"]  = BARRIER_PC;
    return v;
}

static WarpContext* setup_warp(
    SMContext& sm,
    std::vector<ptxemu::ir::StatementContext>& v,
    std::map<std::string, int>& l2pc)
{
    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1,1,1}, b{32,1,1}, bi{0,0,0};
    std::map<std::string, std::unique_ptr<Symtable>> n2s;
    blk->init(g, b, bi, v, &n2s, l2pc);
    blk->sharedMemBytes = 1024;
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    return sm.get_warp(0);
}

// 统计达到某 PC 的 lane 数
static int count_lanes_at_pc(WarpContext* w, int pc) {
    int n = 0;
    for (int i = 0; i < 32; i++) {
        if (w->get_warp_state().threads[i].pc == (uint32_t)pc) n++;
    }
    return n;
}

} // namespace

// =============================================================================
// 测试主入口：simpleGEMM 模式 — divergence + barrier 收敛后, 所有 lane 必须
// 到达 MAIN_LOOP_PC
// =============================================================================
TEST_CASE("I-1: simpleGEMM pattern — both divergent halves reach main loop "
          "after barrier convergence",
          "[barrier][divergence][integration][simplegemm-pattern][BUG-RECONVERGENCE]")
{
    init_factory();

    std::map<std::string, int> l2pc;
    auto v = build_instrs(l2pc);
    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup_warp(sm, v, l2pc);

    // Lanes 16-31 走 PATH_B（短路径直达 barrier）
    // Lanes 0-15 走 PATH_A（长路径，多个 nop 后 bra 到 barrier）
    setup_pred(w, 0xFFFF0000u);

    // Drive scheduler with bounded steps (avoid infinite loop)
    constexpr int MAX_STEPS = 200;
    int steps_taken = 0;
    bool reached_main_loop = false;

    for (int step = 0; step < MAX_STEPS; step++) {
        if (w->is_finished()) {
            reached_main_loop = true;
            steps_taken = step;
            break;
        }
        step_warp(w, v);
        steps_taken = step;
    }

    // 检查所有 32 个 lane 都推进到 MAIN_LOOP_PC（必须）且最终 ret
    int lanes_at_main_loop = count_lanes_at_pc(w, MAIN_LOOP_PC);
    int lanes_at_ret       = count_lanes_at_pc(w, RET_PC);
    int lanes_at_barrier   = count_lanes_at_pc(w, BARRIER_PC);

    INFO("Steps taken: " << steps_taken);
    INFO("Lanes at MAIN_LOOP_PC=" << MAIN_LOOP_PC << ": " << lanes_at_main_loop);
    INFO("Lanes at RET_PC="       << RET_PC       << ": " << lanes_at_ret);
    INFO("Lanes at BARRIER_PC="   << BARRIER_PC   << ": " << lanes_at_barrier);
    INFO("Warp finished: " << w->is_finished());

    // 期望（修复后）：所有 32 个 lane 都能通过 barrier，最终到达 RET_PC。
    // 当前 bug：lanes 0-15 卡在 BARRIER_PC，warp.is_finished() 永不为 true。
    REQUIRE(lanes_at_main_loop + lanes_at_ret >= 16);  // 至少 16 个 lane 到达主循环
    REQUIRE(lanes_at_barrier == 0);                    // 没有 lane 卡在 barrier
    REQUIRE(w->is_finished());                          // warp 必须完成
}

// =============================================================================
// 测试 2: 单独跟踪 lanes 0-15 — 必须执行 ret（不能卡在 barrier）
// =============================================================================
TEST_CASE("I-2: simpleGEMM pattern — lanes 0-15 (the divergent half) "
          "must not be stuck at barrier",
          "[barrier][divergence][integration][simplegemm-pattern][BUG-RECONVERGENCE]")
{
    init_factory();

    std::map<std::string, int> l2pc;
    auto v = build_instrs(l2pc);
    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup_warp(sm, v, l2pc);

    setup_pred(w, 0xFFFF0000u);

    constexpr int MAX_STEPS = 200;
    for (int step = 0; step < MAX_STEPS; step++) {
        if (w->is_finished()) break;
        step_warp(w, v);
    }

    // 重点检查 lanes 0-15 是否到达 RET_PC
    int lanes_0_15_at_or_past_ret = 0;
    int lanes_0_15_stuck = 0;
    int lanes_0_15_at_barrier = 0;
    int lanes_0_15_pc = 0;
    int stuck_lane_idx = -1;
    uint32_t stuck_lane_pc = 0;
    for (int i = 0; i < 16; i++) {
        uint32_t pc = w->get_warp_state().threads[i].pc;
        // After ret handler, PC may advance by 1 (PipelineHandler behavior), so accept
        // pc == RET_PC OR pc == RET_PC + 1 as "reached ret".
        if (pc == (uint32_t)RET_PC || pc == (uint32_t)(RET_PC + 1)) lanes_0_15_at_or_past_ret++;
        else if (pc == (uint32_t)BARRIER_PC) lanes_0_15_at_barrier++;
        else {
            lanes_0_15_pc++;
            stuck_lane_idx = i;
            stuck_lane_pc = pc;
        }
    }
    lanes_0_15_stuck = lanes_0_15_at_barrier;
    UNSCOPED_INFO("Stuck lane idx=" << stuck_lane_idx << " PC=" << stuck_lane_pc);
    INFO("Lanes 0-15 at or past RET_PC=" << RET_PC << ": " << lanes_0_15_at_or_past_ret);
    INFO("Lanes 0-15 stuck at BARRIER_PC=" << BARRIER_PC << ": " << lanes_0_15_at_barrier);

    INFO("Lanes 0-15 at RET_PC=" << RET_PC << ": " << lanes_0_15_at_or_past_ret);
    INFO("Lanes 0-15 stuck at BARRIER_PC=" << BARRIER_PC << ": " << lanes_0_15_at_barrier);
    INFO("Lanes 0-15 at other PCs: " << lanes_0_15_pc);

    // 期望：所有 16 个 lane 0-15 都必须到达 RET_PC（已 ret）。
    REQUIRE(lanes_0_15_at_or_past_ret == 16);
    REQUIRE(lanes_0_15_at_barrier == 0);
}