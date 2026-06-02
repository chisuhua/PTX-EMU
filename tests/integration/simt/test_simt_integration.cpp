#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "memory/resource_manager.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/cta_context.h"
#include "ptx_ir/statement_factory.h"
#include "ptxsim/instruction_factory.h"

using namespace ptxsim;
using namespace ptxir::factory;

static void setup_full_warp(WarpContext& warp) {
    for (int i = 0; i < 32; i++) {
        warp.get_warp_state().threads[i].pc = 10;
        warp.get_warp_state().threads[i].next_pc = 10;
        warp.get_warp_state().threads[i].is_active = true;
        warp.get_warp_state().threads[i].is_exited = false;
        warp.get_warp_state().threads[i].is_blocked = false;
        warp.get_warp_state().threads[i].status = ThreadStatus::Active;
    }
    warp.get_warp_state().exec_mask = 0xFFFFFFFF;
}

TEST_CASE("I1: full divergence-convergence cycle", "[integration]") {
    WarpContext warp;
    setup_full_warp(warp);

    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    for (int i = 0; i < 16; i++) warp.set_thread_pc(i, 20);
    for (int i = 16; i < 32; i++) warp.set_thread_pc(i, 11);

    REQUIRE(warp.get_simt_stack().depth() == 1);

    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 30);
    warp.check_reconvergence();

    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("I2: nested branches with multiple levels", "[integration]") {
    WarpContext warp;
    setup_full_warp(warp);

    SIMTStackEntry l1, l2, l3;
    l1.branch_pc = 10; l1.reconvergence_pc = 50;
    l1.active_mask = 0x0000FFFF; l1.return_mask = 0xFFFFFFFF; l1.return_pc = 50;
    l2.branch_pc = 20; l2.reconvergence_pc = 40;
    l2.active_mask = 0x000000FF; l2.return_mask = 0x0000FFFF; l2.return_pc = 40;
    l3.branch_pc = 25; l3.reconvergence_pc = 35;
    l3.active_mask = 0x0000000F; l3.return_mask = 0x000000FF; l3.return_pc = 35;

    warp.get_simt_stack().push(l1);
    warp.get_simt_stack().push(l2);
    warp.get_simt_stack().push(l3);
    warp.set_exec_mask(0x0000000F);

    REQUIRE(warp.get_simt_stack().depth() == 3);

    for (int i = 0; i < 8; i++) warp.set_thread_pc(i, 35);
    warp.check_reconvergence();
    REQUIRE(warp.get_simt_stack().depth() == 2);
    REQUIRE(warp.get_exec_mask() == 0x000000FF);

    for (int i = 0; i < 16; i++) warp.set_thread_pc(i, 40);
    warp.check_reconvergence();
    REQUIRE(warp.get_simt_stack().depth() == 1);
    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);

    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 50);
    warp.check_reconvergence();
    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("I3: branch + barrier combination", "[integration]") {
    WarpContext warp;
    setup_full_warp(warp);

    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);

    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 30);
    warp.check_reconvergence();

    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("I4: convergence with thread exits", "[integration][exit]") {
    WarpContext warp;
    setup_full_warp(warp);

    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);

    for (int i = 0; i < 8; i++) {
        warp.get_warp_state().threads[i].is_exited = true;
        warp.get_warp_state().threads[i].is_active = false;
        warp.get_warp_state().threads[i].pc = 0;
    }
    for (int i = 8; i < 32; i++) {
        warp.get_warp_state().threads[i].pc = 30;
    }

    warp.check_reconvergence();

    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("I5: scheduler skips unfinished warp", "[integration][scheduler]") {
    WarpContext warp;
    setup_full_warp(warp);

    warp.get_warp_state().threads[0].pc = 10;
    warp.get_warp_state().threads[0].next_pc = 20;
    REQUIRE(warp.is_warp_ready_to_fetch() == false);

    warp.get_warp_state().threads[0].pc = 20;
    REQUIRE(warp.is_warp_ready_to_fetch() == true);
}

TEST_CASE("I6: divergent warp executes one PC group per cycle", "[integration][divergence][cycle_count]") {
    // BUG-SIMT-001: Divergent warps should execute only ONE PC group per cycle.
    // The bug: all PC groups execute in a single cycle.
    // The fix: only the lowest PC group executes per cycle.

    InstructionFactory::initialize();
    ResourceManager::instance().initialize(1, 8192);

    SMContext sm(4, 128, 4096, 0);
    std::unique_ptr<CTAContext> block = std::make_unique<CTAContext>();

    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};

    std::vector<StatementContext> statements;
    statements.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"0"}}},
        "mov.u32 %r1, 0;"));
    statements.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 2}}, OperandContext{ImmOperand{"1"}}},
        "mov.u32 %r2, 1;"));
    statements.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 3}}, OperandContext{ImmOperand{"2"}}},
        "mov.u32 %r3, 2;"));
    statements.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 4}}, OperandContext{ImmOperand{"3"}}},
        "mov.u32 %r4, 3;"));
    statements.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 5}}, OperandContext{ImmOperand{"4"}}},
        "mov.u32 %r5, 4;"));

    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;
    block->init(gridDim, blockDim, blockIdx, statements, &name2Sym, label2pc);

    sm.add_block(std::move(block));

    WarpContext* warp = sm.get_warp(0);
    REQUIRE(warp != nullptr);

    for (int i = 0; i < 32; i++) {
        warp->advance_thread_pc(i, 0);
    }

    auto lanes_by_pc_before = warp->get_lanes_by_pc();
    int pc_groups_before = static_cast<int>(lanes_by_pc_before.size());

    sm.exe_once();

    auto lanes_by_pc_after = warp->get_lanes_by_pc();
    int pc_groups_after = static_cast<int>(lanes_by_pc_after.size());

    CHECK(pc_groups_before == 1);
    CHECK(pc_groups_after == 1);
    CHECK(warp->get_warp_state().threads[0].pc == 1);
}
