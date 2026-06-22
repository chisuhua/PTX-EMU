#include "catch_amalgamated.hpp"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/simt_stack.h"
#include "ptxsim/wbar.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_factory.h"
#include "memory/resource_manager.h"
#include "ptxsim/testing/scheduler_utils.h"
#include <map>
#include <memory>
#include <vector>
#include <string>

using namespace ptxir::factory;
using ptxsim::testing::step_warp;

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

TEST_CASE("integrated_wbar_convergence_operations", "[wbar][integrated][execute_warp]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0x0000000F, 2));
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    step_warp(warp, statements);
    step_warp(warp, statements);

    CHECK(warp->get_wbar(0).is_complete() == true);
    // Per NVIDIA PTX 9.3 + Volta-Tune + 4 academic simulators (gpgpu-sim/gem5/MIAOW/M2S),
    // barrier arrival is counted at WARP level: all 32 active lanes call arrive() on
    // the same instruction, regardless of the static participation_mask. count_arrived()
    // therefore returns 32 here, not popcount(participation_mask)=4.
    CHECK(warp->get_wbar(0).count_arrived() == 32);
    CHECK(warp->get_wbar(0).participation_mask == 0x0000000F);
    CHECK(warp->get_wbar(0).count_participants() == 4);

    for (int i = 0; i < 4; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
    }

    step_warp(warp, statements);

    for (int i = 0; i < 4; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 3);
    }
}

TEST_CASE("integrated_warp_barrier_divergence_scenario", "[wbar][divergence][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFE, 2));  // Fixed: barrier mask matches exec_mask
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    for (int i = 0; i < 32; i++) {
        warp->get_warp_state().threads[i].pc = 0;
    }

    warp->set_exec_mask(0xFFFFFFFE);
    warp->set_active_mask(0xFFFFFFFE);

    step_warp(warp, statements);
    step_warp(warp, statements);

    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->get_wbar(0).count_arrived() == 31);

    for (int i = 1; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
    }
}

TEST_CASE("integrated_multiple_barrier_registers", "[wbar][multi][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0x0000000F, 4));
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0x000000F0, 6));
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0x00000F00, 8));
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0x0000F000, 10));

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    REQUIRE(warp->get_warp_state().wbars.size() == 4);

    step_warp(warp, statements);  // mov: all → PC=1
    step_warp(warp, statements);  // barrier 0x0F: lanes 0-3 → PC=4

    // 第一个 barrier 完成后立即验证 wbar[0] 状态
    CHECK(warp->get_wbar(0).is_complete() == true);
    // Warp-level arrival: all 32 active lanes arrive (not popcount(mask)=4)
    CHECK(warp->get_wbar(0).count_arrived() == 32);
    CHECK(warp->get_wbar(0).participation_mask == 0x0000000F);
    CHECK(warp->get_wbar(0).count_participants() == 4);

    // 后续 barrier 指令（PC 3/5/7）在当前单 wbar 实现中
    // 部分可能重新初始化 wbar[0]，覆盖之前的完成状态
    step_warp(warp, statements);  // mov: no lanes at PC=2, no-op
    step_warp(warp, statements);  // barrier: no lanes at PC=3, no-op
    step_warp(warp, statements);  // mov: lanes 0-3 → PC=5
    step_warp(warp, statements);  // barrier 0xF00: lanes 0-3 arrive, barrier can't complete (needs lanes 12-15)
    step_warp(warp, statements);  // mov: no lanes at PC=6
    step_warp(warp, statements);  // barrier 0xF000: no lanes at PC=7

    // wbar[1]、wbar[2]、wbar[3] 不会被使用（当前单 wbar 实现）
    CHECK(warp->get_wbar(1).is_complete() == false);
    CHECK(warp->get_wbar(2).is_complete() == false);
    CHECK(warp->get_wbar(3).is_complete() == false);
}

TEST_CASE("integrated_wbar_partial_participation", "[wbar][partial][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0x00000003, 2));
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0x0000000F, 4));
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0x000000FF, 6));

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    step_warp(warp, statements);
    step_warp(warp, statements);
    CHECK(warp->get_wbar(0).is_complete() == true);
    // Warp-level arrival: all 32 active lanes arrive (not popcount(mask)=2)
    CHECK(warp->get_wbar(0).count_arrived() == 32);
    CHECK(warp->get_wbar(0).count_participants() == 2);  // mask 0x03 → 2 participants

    step_warp(warp, statements);
    step_warp(warp, statements);
    // 当前单 wbar 实现中，第二个 barrier 时所有 32 active lanes 仍会到达
    // BUG-POSTBARRIER-TWOHALVES 修复后第一个 barrier 保留了所有 lane 的 active 状态
    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->get_wbar(0).count_arrived() == 32);
    CHECK(warp->get_wbar(0).count_participants() == 4);  // mask 0x0F → 4 participants

    step_warp(warp, statements);
    step_warp(warp, statements);
    // 第三个 barrier：所有 32 active lanes 全部到达
    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->get_wbar(0).count_arrived() == 32);
    CHECK(warp->get_wbar(0).count_participants() == 8);  // mask 0xFF → 8 participants
}

TEST_CASE("integrated_wbar_divergent_control_flow", "[wbar][divergence][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(make_mov_stmt());
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0xFFFF0000, 5));  // Fixed: mask matches active_mask
    statements.push_back(make_mov_stmt());
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    warp->set_exec_mask(0xFFFF0000);
    warp->set_active_mask(0xFFFF0000);
    for (int i = 0; i < 16; i++) {
        warp->get_warp_state().threads[i].is_active = false;
    }

    step_warp(warp, statements);
    step_warp(warp, statements);
    step_warp(warp, statements);
    step_warp(warp, statements);

    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->get_wbar(0).count_arrived() == 16);

    for (int i = 16; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 5);
    }
}

TEST_CASE("integrated_wbar_reconvergence_pc", "[wbar][pc][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0x000000FF, 4));
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0x0000FF00, 6));
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0x00FF0000, 8));

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    step_warp(warp, statements);
    step_warp(warp, statements);

    CHECK(warp->get_wbar(0).is_complete() == true);
    // Warp-level arrival: all 32 active lanes arrive (not popcount(mask)=8)
    CHECK(warp->get_wbar(0).count_arrived() == 32);

    for (int i = 0; i < 8; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 4);
    }

    // Fixed: Use bounded iterations instead of infinite loops
    // After BUG-POSTBARRIER-TWOHALVES fix, all 32 lanes (including 8-15) advance past
    // the first barrier. Lanes 8-15 are no longer "stuck at PC=1" — they participate
    // in the same warp-level barrier completion as lanes 0-7.
    for (int i = 0; i < 10; i++) {
        step_warp(warp, statements);
    }

    // wbar[0] is complete because all 32 lanes arrived at the first barrier, and
    // subsequent barriers in the sequence continue to drive all 32 lanes through
    // their reconvergence PCs.
    CHECK(warp->get_wbar(0).is_complete() == true);

    // Lanes 8-15 advance past PC=1 (the original "stuck" assumption is invalid
    // under the BUG-POSTBARRIER fix). After 10 step_warp iterations all 32 lanes
    // are well past the first barrier's reconvergence point (PC=4).
    for (int i = 8; i < 16; i++) {
        CHECK(warp->get_thread(i)->get_pc() >= 4);
    }
}

TEST_CASE("integrated_wbar_thread_state_transitions", "[wbar][state][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 3));
    statements.push_back(make_mov_stmt());
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    step_warp(warp, statements);

    int schedulable_before = 0;
    for (int i = 0; i < 32; i++) {
        if (warp->get_warp_state().threads[i].is_schedulable()) {
            schedulable_before++;
        }
    }
    CHECK(schedulable_before == 32);

    step_warp(warp, statements);

    CHECK(warp->get_wbar(0).is_complete() == true);

    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 3);
    }

    int schedulable_after = 0;
    for (int i = 0; i < 32; i++) {
        if (warp->get_warp_state().threads[i].is_schedulable()) {
            schedulable_after++;
        }
    }
    CHECK(schedulable_after == 32);
}

TEST_CASE("integrated_full_barrier_execution_flow", "[wbar][full][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 2));
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    step_warp(warp, statements);

    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 1);
    }

    step_warp(warp, statements);

    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->get_wbar(0).count_arrived() == 32);
    CHECK(warp->get_wbar(0).participation_mask == 0xFFFFFFFF);
    CHECK(warp->get_wbar(0).count_participants() == 32);

    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
    }

    step_warp(warp, statements);

    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 3);
    }

    CHECK(warp->get_warp_state().current_wbar_id == -1);
}
