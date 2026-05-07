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
#include "memory/resource_manager.h"
#include <map>
#include <memory>
#include <vector>
#include <string>

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

static StatementContext make_barrier_stmt(uint32_t mask, int reconvergence_pc) {
    StatementContext ctx;
    ctx.type = S_BAR_WARP_SYNC;
    BarWarpSyncInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(mask)}});
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(reconvergence_pc)}});
    ctx.data = instr;
    ctx.instructionText = "bar.warp.sync.b32 " + std::to_string(mask) + ", " +
                          std::to_string(reconvergence_pc) + ";";
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
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;
    block->init(gridDim, blockDim, blockIdx, statements, &name2Sym, label2pc);
    return block;
}

TEST_CASE("integrated_wbar_convergence_operations", "[wbar][integrated][execute_warp]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(make_barrier_stmt(0x0000000F, 2));
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    warp->execute_warp_instruction(statements[0], 0);
    warp->execute_warp_instruction(statements[1], 1);

    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->get_wbar(0).count_arrived() == 4);
    CHECK(warp->get_wbar(0).participation_mask == 0x0000000F);
    CHECK(warp->get_wbar(0).count_participants() == 4);

    for (int i = 0; i < 4; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
    }

    warp->execute_warp_instruction(statements[2], 2);

    for (int i = 0; i < 4; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 3);
    }
}

TEST_CASE("integrated_warp_barrier_divergence_scenario", "[wbar][divergence][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(make_mov_stmt());
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    for (int i = 0; i < 32; i++) {
        warp->get_warp_state().threads[i].pc = 0;
    }

    warp->set_exec_mask(0xFFFFFFFE);
    warp->set_active_mask(0xFFFFFFFE);

    warp->execute_warp_instruction(statements[0], 0);

    statements[1] = make_barrier_stmt(0xFFFFFFFE, 2);
    warp->execute_warp_instruction(statements[1], 1);

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
    statements.push_back(make_barrier_stmt(0x0000000F, 4));
    statements.push_back(make_mov_stmt());
    statements.push_back(make_barrier_stmt(0x000000F0, 6));
    statements.push_back(make_mov_stmt());
    statements.push_back(make_barrier_stmt(0x00000F00, 8));
    statements.push_back(make_mov_stmt());
    statements.push_back(make_barrier_stmt(0x0000F000, 10));

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    REQUIRE(warp->get_warp_state().wbars.size() == 4);

    warp->execute_warp_instruction(statements[0], 0);
    warp->execute_warp_instruction(statements[1], 1);
    warp->execute_warp_instruction(statements[2], 2);
    warp->execute_warp_instruction(statements[3], 3);
    warp->execute_warp_instruction(statements[4], 4);
    warp->execute_warp_instruction(statements[5], 5);
    warp->execute_warp_instruction(statements[6], 6);
    warp->execute_warp_instruction(statements[7], 7);

    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->get_wbar(1).is_complete() == true);
    CHECK(warp->get_wbar(2).is_complete() == true);
    CHECK(warp->get_wbar(3).is_complete() == true);

    CHECK(warp->get_wbar(0).count_arrived() == 4);
    CHECK(warp->get_wbar(1).count_arrived() == 4);
    CHECK(warp->get_wbar(2).count_arrived() == 4);
    CHECK(warp->get_wbar(3).count_arrived() == 4);
}

TEST_CASE("integrated_wbar_partial_participation", "[wbar][partial][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(make_barrier_stmt(0x00000003, 2));
    statements.push_back(make_mov_stmt());
    statements.push_back(make_barrier_stmt(0x0000000F, 4));
    statements.push_back(make_mov_stmt());
    statements.push_back(make_barrier_stmt(0x000000FF, 6));

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    warp->execute_warp_instruction(statements[0], 0);
    warp->execute_warp_instruction(statements[1], 1);
    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->get_wbar(0).count_arrived() == 2);
    CHECK(warp->get_wbar(0).count_participants() == 2);

    warp->execute_warp_instruction(statements[2], 2);
    warp->execute_warp_instruction(statements[3], 3);
    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->get_wbar(0).count_arrived() == 4);
    CHECK(warp->get_wbar(0).count_participants() == 4);

    warp->execute_warp_instruction(statements[4], 4);
    warp->execute_warp_instruction(statements[5], 5);
    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->get_wbar(0).count_arrived() == 8);
    CHECK(warp->get_wbar(0).count_participants() == 8);
}

TEST_CASE("integrated_wbar_divergent_control_flow", "[wbar][divergence][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(make_mov_stmt());
    statements.push_back(make_mov_stmt());
    statements.push_back(make_barrier_stmt(0xFFFFFFFF, 5));
    statements.push_back(make_mov_stmt());
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    warp->set_exec_mask(0xFFFF0000);
    warp->set_active_mask(0xFFFF0000);
    for (int i = 0; i < 16; i++) {
        warp->get_warp_state().threads[i].is_active = false;
    }

    warp->execute_warp_instruction(statements[0], 0);
    warp->execute_warp_instruction(statements[1], 1);
    warp->execute_warp_instruction(statements[2], 2);
    warp->execute_warp_instruction(statements[3], 3);

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
    statements.push_back(make_barrier_stmt(0x000000FF, 4));
    statements.push_back(make_mov_stmt());
    statements.push_back(make_barrier_stmt(0x0000FF00, 6));
    statements.push_back(make_mov_stmt());
    statements.push_back(make_barrier_stmt(0x00FF0000, 8));

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    warp->execute_warp_instruction(statements[0], 0);
    warp->execute_warp_instruction(statements[1], 1);

    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->get_wbar(0).count_arrived() == 8);

    for (int i = 0; i < 8; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 4);
    }

    warp->execute_warp_instruction(statements[2], 2);
    warp->execute_warp_instruction(statements[3], 3);

    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->get_wbar(0).count_arrived() == 8);

    for (int i = 8; i < 16; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 6);
    }
}

TEST_CASE("integrated_wbar_thread_state_transitions", "[wbar][state][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(make_barrier_stmt(0xFFFFFFFF, 3));
    statements.push_back(make_mov_stmt());
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    warp->execute_warp_instruction(statements[0], 0);

    int schedulable_before = 0;
    for (int i = 0; i < 32; i++) {
        if (warp->get_warp_state().threads[i].is_schedulable()) {
            schedulable_before++;
        }
    }
    CHECK(schedulable_before == 32);

    warp->execute_warp_instruction(statements[1], 1);

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
    statements.push_back(make_barrier_stmt(0xFFFFFFFF, 2));
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    warp->execute_warp_instruction(statements[0], 0);

    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 1);
    }

    warp->execute_warp_instruction(statements[1], 1);

    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->get_wbar(0).count_arrived() == 32);
    CHECK(warp->get_wbar(0).participation_mask == 0xFFFFFFFF);
    CHECK(warp->get_wbar(0).count_participants() == 32);

    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
    }

    warp->execute_warp_instruction(statements[2], 2);

    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 3);
    }

    CHECK(warp->get_warp_state().current_wbar_id == -1);
}
