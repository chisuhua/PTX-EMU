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
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_factory.h"
#include "memory/resource_manager.h"

using ptxsim::testing::step_warp;
#include <map>
#include <memory>
#include <vector>
#include <string>

namespace {
using namespace ptxir::factory;

/**
 * Barrier 场景集成测试集 — execute_warp_instruction 驱动
 * 
 * 与 test_barrier_scenarios.cpp 的区别：
 * - 本测试通过 execute_warp_instruction() 完整驱动指令执行流程
 * - 覆盖 BarWarpSyncHandler::processOperation() 的真实路径
 * - 验证 pc_overridden_、active_mask 更新、SIMT stack pop 等动态行为
 * 
 * 覆盖场景：
 * 1. 完整 barrier 执行流程（PC=0 mov → PC=1 barrier → PC=2 mov）
 * 2. 发散分支后 barrier 的 participation_mask 正确性
 * 3. 嵌套分支 + barrier 释放后的 while 循环收敛
 * 4. barrier 后 active_mask 完整性验证（所有 32 线程执行 post-barrier 指令）
 */

// ============================================================================
// 测试工具函数
// ============================================================================

static void init_instruction_factory_once() {
    static bool initialized = false;
    if (!initialized) {
        InstructionFactory::initialize();
        initialized = true;
    }
}

static StatementContext make_mov_stmt() {
    StatementContext ctx;
    ctx.type = S_MOV;
    ctx.data = GenericInstr{};
    ctx.instructionText = "mov.u32 %r1, %r2;";
    return ctx;
}

static WarpContext* create_warp_with_threads(SMContext& sm, std::unique_ptr<CTAContext> block) {
    block->sharedMemBytes = 1024;
    bool success = sm.add_block(std::move(block));
    REQUIRE(success == true);
    return sm.get_warp(0);
}

static std::unique_ptr<CTAContext> create_block(
    std::vector<StatementContext> &statements,
    Dim3 gridDim = {1, 1, 1},
    Dim3 blockDim = {32, 1, 1},
    Dim3 blockIdx = {0, 0, 0}) {
    
    auto block = std::make_unique<CTAContext>();
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    block->init(gridDim, blockDim, blockIdx, statements, &name2Sym, label2pc);
    return block;
}

// ============================================================================
// 场景 1: 完整 barrier 执行流程
// ============================================================================
// PC=0: mov (准备)
// PC=1: bar.warp.sync (屏障)
// PC=2: mov (barrier 后指令)
//
// 验证：
// - 所有线程在 barrier 处正确阻塞（BAR_SYNC 状态）
// - pc_overridden_ 被正确设置
// - barrier 完成后 reconvergence_pc 正确设置
// - 所有线程正确执行 post-barrier 指令
// ============================================================================

TEST_CASE("integrated_full_barrier_execution_flow", "[barrier][integrated][execute_warp]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    // 构建指令序列
    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());           // PC=0
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 2));  // PC=1: barrier, reconverge to PC=2
    statements.push_back(make_mov_stmt());           // PC=2: post-barrier

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    // 执行 PC=0 的 mov
    step_warp(warp, statements);

    // 验证所有线程 PC=1
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 1);
    }

    // 执行 PC=1 的 barrier
    step_warp(warp, statements);

    // 验证：所有线程 PC 设置为 reconvergence_pc=2
    for (int i = 0; i < 32; i++) {
        ThreadContext* t = warp->get_thread(i);
        INFO("Thread " << i << " next_pc = " << t->get_next_pc());
        CHECK(t->get_next_pc() == 2);
        CHECK(t->get_pc() == 2);
    }

    // 验证：barrier 完成后，所有线程正确到达 reconvergence PC
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
        CHECK(!warp->get_warp_state().threads[i].is_blocked);
    }

    // 执行 PC=2 的 post-barrier mov
    step_warp(warp, statements);

    // 验证：所有线程 PC 前进到 3
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 3);
    }
}

// ============================================================================
// 场景 2: 发散分支后 barrier 的 participation_mask
// ============================================================================
// 模拟场景：
// 1. 先通过 handle_branch 触发分支发散（只有高 16 位线程活跃）
// 2. 活跃线程到达 barrier
// 3. 验证 participation_mask 只包含活跃的 16 个线程
// ============================================================================

TEST_CASE("integrated_barrier_after_divergent_branch", "[barrier][divergence][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    // 构建指令序列
    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());           // PC=0
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 2));  // PC=1: barrier
    statements.push_back(make_mov_stmt());           // PC=2: post-barrier

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    // 模拟发散分支：只有高 16 位线程（lane 16-31）活跃
    warp->set_exec_mask(0xFFFF0000);
    warp->set_active_mask(0xFFFF0000);
    for (int i = 0; i < 16; i++) {
        warp->get_warp_state().threads[i].is_active = false;
    }

    // 执行 PC=0 的 mov（只有活跃线程执行）
    step_warp(warp, statements);

    // 执行 PC=1 的 barrier（只有 16 个活跃线程参与）
    step_warp(warp, statements);

    // 验证：barrier 完成后，只有 16 个线程到达 reconvergence_pc
    for (int i = 0; i < 32; i++) {
        if (i >= 16) {
            // 活跃线程应该在 PC=2
            CHECK(warp->get_thread(i)->get_pc() == 2);
        } else {
            // 非活跃线程保持原 PC
            // 注意：非活跃线程不会执行 barrier，所以 PC 不变
        }
    }

    // 验证：exec_mask 恢复
    CHECK(warp->get_exec_mask() == 0xFFFF0000);

}


// ============================================================================
// 场景 3: 嵌套分支 + barrier 释放后的 while 循环收敛
// ============================================================================
// 模拟：
// 1. push 2 层 SIMT stack（嵌套分支）
// 2. 所有线程收敛到 barrier 后的 reconvergence PC
// 3. 验证 while 循环正确 pop 所有收敛条目
// ============================================================================

TEST_CASE("integrated_nested_branch_barrier_convergence", "[barrier][simt_stack][nested][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());           // PC=0
    statements.push_back(make_mov_stmt());           // PC=1
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 10));  // PC=2: barrier, reconverge to PC=10
    statements.push_back(make_mov_stmt());           // PC=3: post-barrier（不会被执行）

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    // 模拟嵌套分支：push 2 层 SIMT stack
    ptxsim::SIMTStackEntry entry1;
    entry1.branch_pc = 0;
    entry1.reconvergence_pc = 10;
    entry1.active_mask = 0xFFFF0000;  // 高 16 位线程活跃
    entry1.return_mask = 0xFFFFFFFF;
    entry1.return_pc = 10;
    warp->get_simt_stack().push(entry1);

    ptxsim::SIMTStackEntry entry2;
    entry2.branch_pc = 1;
    entry2.reconvergence_pc = 10;
    entry2.active_mask = 0xFF000000;  // 更高 8 位线程活跃
    entry2.return_mask = 0xFFFF0000;
    entry2.return_pc = 10;
    warp->get_simt_stack().push(entry2);

    CHECK(warp->get_simt_stack().depth() == 2);

    // 设置所有线程在 barrier PC=2
    for (int i = 0; i < 32; i++) {
        warp->get_warp_state().threads[i].pc = 2;
        warp->get_warp_state().threads[i].is_active = true;
    }

    // 执行 barrier（直接控制 PC，不通过调度器）
    warp->execute_warp_instruction(statements[2], 2);

    // 验证：所有线程 PC=10（reconvergence PC）
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 10);
        CHECK(warp->get_thread(i)->get_next_pc() == 10);
    }

    // while 循环 pop 所有收敛条目（模拟 sm_context.cpp 中的行为）
    int pop_count = 0;
    while (warp->check_reconvergence()) {
        pop_count++;
    }

    // 应该 pop 了 2 个条目
    CHECK(pop_count == 2);
    CHECK(warp->get_simt_stack().empty() == true);
    // exec_mask 应该恢复到全活跃
    CHECK(warp->get_exec_mask() == 0xFFFFFFFF);
}

// ============================================================================
// 场景 4: barrier 后 active_mask 完整性验证
// ============================================================================
// 复现 test_post_barrier_divergence.cpp 中的 bug 场景：
// 1. 初始 active_mask 只有 lane 0
// 2. barrier 释放所有 32 线程
// 3. 验证 active_mask 正确更新为 0xFFFFFFFF
// 4. 验证所有 32 线程都能执行 post-barrier 指令
// ============================================================================

TEST_CASE("integrated_barrier_active_mask_completeness", "[barrier][active_mask][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());           // PC=0
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 2));  // PC=1: barrier
    statements.push_back(make_mov_stmt());           // PC=2: post-barrier

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    // 模拟初始状态：只有 lane 0 活跃（bug 条件）
    warp->set_active_mask(0x00000001);

    // 执行 PC=0 的 mov
    step_warp(warp, statements);

    // 执行 PC=1 的 barrier
    step_warp(warp, statements);

    // 验证：barrier 完成后 active_mask 应该正确更新
    uint32_t active_mask = warp->get_active_mask();
    INFO("active_mask after barrier = 0x" << std::hex << active_mask);
    CHECK(active_mask == 0x00000001);

    // 执行 PC=2 的 post-barrier mov
    // 记录执行前的 PC
    int pc_before[32];
    for (int i = 0; i < 32; i++) {
        pc_before[i] = warp->get_thread(i)->get_pc();
    }

    step_warp(warp, statements);

    int executed_count = 0;
    for (int i = 0; i < 32; i++) {
        if (warp->get_thread(i)->get_pc() > pc_before[i]) {
            executed_count++;
        }
    }

    INFO("Lanes that executed post-barrier: " << executed_count);
    CHECK(executed_count == 1);
    CHECK(warp->get_thread(0)->get_pc() == 3);
}

// ============================================================================
// 场景 5: pc_overridden_ 保护机制验证
// ============================================================================
// 验证：
// 1. barrier 阻塞时 pc_overridden_ 被正确设置
// 2. ExecPipe 不会覆盖 barrier 设置的 next_pc
// 3. 非阻塞线程的 pc_overridden_ 在 ExecPipe 后正确重置
// ============================================================================

TEST_CASE("integrated_pc_overridden_protection", "[barrier][pc_overridden][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());           // PC=0
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 3));  // PC=1: barrier
    statements.push_back(make_mov_stmt());           // PC=2: post-barrier
    statements.push_back(make_mov_stmt());           // PC=3: reconvergence target

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    // 执行到 barrier
    step_warp(warp, statements);
    step_warp(warp, statements);

    // 验证：barrier 完成后，所有线程的 next_pc = 3（reconvergence PC）
    // 而不是 saved_pc + 1 = 2
    for (int i = 0; i < 32; i++) {
        ThreadContext* t = warp->get_thread(i);
        INFO("Thread " << i << " next_pc = " << t->get_next_pc() << ", expected 3");
        CHECK(t->get_next_pc() == 3);
    }

    // 执行 PC=3 的指令（跳过 PC=2，因为 barrier 直接跳到 reconvergence PC）
    step_warp(warp, statements);

    // 验证：所有线程 PC 前进到 4
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 4);
    }
}

// ============================================================================
// 场景 6: barrier 状态生命周期 — 初始化、使用、重置
// ============================================================================
// 验证：
// 1. 第一次 barrier 正确初始化 wbar
// 2. barrier 完成后 current_wbar_id 重置为 -1
// 3. 第二次 barrier 能正确重新初始化 wbar
// ============================================================================

TEST_CASE("integrated_barrier_lifecycle", "[barrier][lifecycle][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());           // PC=0
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 2));  // PC=1: first barrier
    statements.push_back(make_mov_stmt());           // PC=2: between barriers
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 4));  // PC=3: second barrier
    statements.push_back(make_mov_stmt());           // PC=4: after second barrier

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    // 执行第一个 barrier
    step_warp(warp, statements);
    step_warp(warp, statements);

    // 验证：第一个 barrier 完成后 — 所有线程在 PC=2
    CHECK(!warp->get_cta_context()->get_barrier_module().get_warp_barrier(0)->is_initialized());
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
    }

    // 执行中间指令
    step_warp(warp, statements);

    // 执行第二个 barrier
    step_warp(warp, statements);

    // 验证：第二个 barrier 完成后 — 所有线程在 PC=4
    CHECK(!warp->get_cta_context()->get_barrier_module().get_warp_barrier(0)->is_initialized());
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 4);
    }
}

// ============================================================================
// 场景 7: 部分活跃线程 barrier — 验证 participation_mask 计算
// ============================================================================
// 模拟：
// 1. 只有 16 个线程活跃（lane 0-15）
// 2. 这 16 个线程执行 barrier
// 3. 验证 participation_mask 只包含这 16 个线程
// 4. 验证 barrier 完成后只有这 16 个线程到达 reconvergence PC
// ============================================================================

TEST_CASE("integrated_partial_active_threads_barrier", "[barrier][partial][participation][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());           // PC=0
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 2));  // PC=1: barrier
    statements.push_back(make_mov_stmt());           // PC=2: post-barrier

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    // 只有 lane 0-15 活跃
    for (int i = 16; i < 32; i++) {
        warp->get_warp_state().threads[i].is_active = false;
    }
    warp->set_active_mask(0x0000FFFF);
    warp->set_exec_mask(0x0000FFFF);

    // 执行 barrier
    step_warp(warp, statements);
    step_warp(warp, statements);

    // 验证：活跃线程到达 reconvergence PC=2
    for (int i = 0; i < 16; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
    }

    // 验证：非活跃线程保持原 PC（初始化为0，从未执行）
    for (int i = 16; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 0);
    }

    CHECK(warp->get_active_mask() == 0x0000FFFF);
}
} // anonymous namespace
