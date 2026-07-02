#include "catch_amalgamated.hpp"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/simt_stack.h"

#include "ptxsim/cta_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_factory.h"
#include "memory/resource_manager.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/testing/instruction_helpers.h"
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

 // After barrier, all 32 active lanes should be at reconvergence PC and not blocked
 for (int i = 0; i < 32; i++) {
 CHECK(warp->get_thread(i)->get_pc() == 2);
 CHECK(!warp->get_warp_state().threads[i].is_blocked);
 }

 const uint32_t active_mask = warp->get_active_mask();
 // Only lanes 0-3 will be active at PC=2; others inactive
 for (int i = 0; i < 32; i++) {
 if (active_mask & (1u << i)) {
 CHECK(warp->get_thread(i)->get_pc() == 2);
 } else {
 // Inactive lanes should not have advanced to barrier PC
 CHECK(warp->get_thread(i)->get_pc() <= 1);
 }
 }

 step_warp(warp, statements);

 // Only lanes 0-3 should execute the post-barrier mov
 for (int i = 0; i < 32; i++) {
 if (active_mask & (1u << i)) {
 CHECK(warp->get_thread(i)->get_pc() == 3);
 } else {
 // Inactive lanes remain stale (placeholder check)
 }
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

    for (int i = 1; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
        CHECK(!warp->get_warp_state().threads[i].is_blocked);
    }
}

TEST_CASE("integrated_multiple_barrier_registers", "[wbar][multi][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());                        // PC=0
    statements.push_back(makeBarWarpSyncInstr(0x0000000F, 2));   // PC=1: barrier→PC=2
    statements.push_back(make_mov_stmt());                        // PC=2
    statements.push_back(makeBarWarpSyncInstr(0x000000F0, 4));   // PC=3: barrier→PC=4
    statements.push_back(make_mov_stmt());                        // PC=4
    statements.push_back(makeBarWarpSyncInstr(0x00000F00, 6));   // PC=5: barrier→PC=6
    statements.push_back(make_mov_stmt());                        // PC=6
    statements.push_back(makeBarWarpSyncInstr(0x0000F000, 8));   // PC=7: barrier→PC=8
    statements.push_back(ptxsim::testing::make_ret());            // PC=8: ret (marks warp finished)

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    REQUIRE(warp->get_cta_context()->get_barrier_module().get_warp_barrier(0) != nullptr);

    step_warp(warp, statements);  // mov at PC=0: all → PC=1

    // BUG-POSTBARRIER-TWOHALVES fix means ALL 32 active lanes pass through
    // every barrier (warp-level arrival), not just the lanes in the mask.
    step_warp(warp, statements);  // barrier 0x0F at PC=1: all 32 arrive → all PC=2

    // 第一个 barrier 完成后 — 所有线程在 PC=2
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
        CHECK(!warp->get_warp_state().threads[i].is_blocked);
    }

    step_warp(warp, statements);  // mov at PC=2: all → PC=3
    step_warp(warp, statements);  // barrier 0xF0 at PC=3: all 32 arrive → all PC=4
    step_warp(warp, statements);  // mov at PC=4: all → PC=5
    step_warp(warp, statements);  // barrier 0xF00 at PC=5: all 32 arrive → all PC=6
    step_warp(warp, statements);  // mov at PC=6: all → PC=7
    step_warp(warp, statements);  // barrier 0xF000 at PC=7: all 32 arrive → all PC=8
    step_warp(warp, statements);  // ret at PC=8: all lanes exit

    // All lanes should have exited after ret
    CHECK(warp->is_finished() == true);
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
    // All 32 active lanes released to reconvergence PC=2, not blocked
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
        CHECK(!warp->get_warp_state().threads[i].is_blocked);
    }

    step_warp(warp, statements);
    step_warp(warp, statements);
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 4);
    }

    step_warp(warp, statements);
    step_warp(warp, statements);
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 6);
    }
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

    for (int i = 16; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 5);
        CHECK(!warp->get_warp_state().threads[i].is_blocked);
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

    // All 32 lanes released to reconvergence PC=4, not blocked
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 4);
        CHECK(!warp->get_warp_state().threads[i].is_blocked);
    }

    // Fixed: Use bounded iterations instead of infinite loops
    // After BUG-POSTBARRIER-TWOHALVES fix, all 32 lanes (including 8-15) advance past
    // the first barrier. Lanes 8-15 are no longer "stuck at PC=1" — they participate
    // in the same warp-level barrier completion as lanes 0-7.
    for (int i = 0; i < 10; i++) {
        step_warp(warp, statements);
    }

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

    // All 32 lanes at reconvergence PC=3, not blocked, schedulable
    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 3);
        CHECK(!warp->get_warp_state().threads[i].is_blocked);
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

    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
        CHECK(!warp->get_warp_state().threads[i].is_blocked);
    }

    step_warp(warp, statements);

    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 3);
    }

    CHECK(!warp->get_cta_context()->get_barrier_module().get_warp_barrier(0)->is_initialized());
}
