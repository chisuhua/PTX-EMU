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
#include <map>
#include <memory>
#include <vector>
#include <string>

using namespace ptxsim;

namespace {
using namespace ptxir::factory;

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

static StatementContext make_nop_stmt() {
    StatementContext ctx;
    ctx.type = S_MOV;
    ctx.data = GenericInstr{};
    ctx.instructionText = "mov.u32 %r1, %r1;";
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

}

TEST_CASE("ShortestFirst mode schedules shorter path first", "[shortest_first][divergence]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    SMContext sm(4, 128, 65536, 0);
    sm.set_divergence_execution_mode(ptxsim::DivergenceExecutionMode::ShortestFirst);

    REQUIRE(sm.get_divergence_execution_mode() == ptxsim::DivergenceExecutionMode::ShortestFirst);

    std::vector<StatementContext> statements;
    for (int i = 0; i < 11; i++) {
        statements.push_back(make_nop_stmt());
    }

    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    for (int i = 0; i < 32; i++) {
        warp->get_warp_state().threads[i].pc = 0;
        warp->get_warp_state().threads[i].next_pc = 0;
        warp->get_warp_state().threads[i].is_active = true;
        warp->get_warp_state().threads[i].is_exited = false;
    }

    warp->set_exec_mask(0xFFFFFFFF);

    SIMTStackEntry entry;
    entry.branch_pc = 0;
    entry.reconvergence_pc = 10;
    entry.active_mask = 0xFFFFFFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 10;
    warp->get_simt_stack().push(entry);

    REQUIRE(warp->get_simt_stack().depth() == 1);
}

TEST_CASE("ShortestFirst mode uses path length heuristic", "[shortest_first][path_length]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    SMContext sm(4, 128, 65536, 0);
    sm.set_divergence_execution_mode(ptxsim::DivergenceExecutionMode::ShortestFirst);

    std::vector<StatementContext> statements;
    for (int i = 0; i < 15; i++) {
        statements.push_back(make_nop_stmt());
    }

    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    for (int i = 0; i < 32; i++) {
        warp->get_warp_state().threads[i].pc = 0;
        warp->get_warp_state().threads[i].next_pc = 0;
        warp->get_warp_state().threads[i].is_active = true;
        warp->get_warp_state().threads[i].is_exited = false;
    }

    warp->set_exec_mask(0xFFFFFFFF);

    SIMTStackEntry entry;
    entry.branch_pc = 0;
    entry.reconvergence_pc = 14;
    entry.active_mask = 0xFFFFFFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 14;
    warp->get_simt_stack().push(entry);

    REQUIRE(warp->get_simt_stack().depth() == 1);
    REQUIRE(warp->get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("ShortestFirst mode reconverges correctly", "[shortest_first][reconvergence]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    SMContext sm(4, 128, 65536, 0);
    sm.set_divergence_execution_mode(ptxsim::DivergenceExecutionMode::ShortestFirst);

    std::vector<StatementContext> statements;
    for (int i = 0; i < 4; i++) {
        statements.push_back(make_nop_stmt());
    }
    statements.push_back(make_mov_stmt());

    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    for (int i = 0; i < 32; i++) {
        warp->get_warp_state().threads[i].pc = 0;
        warp->get_warp_state().threads[i].next_pc = 0;
        warp->get_warp_state().threads[i].is_active = true;
        warp->get_warp_state().threads[i].is_exited = false;
    }

    warp->set_exec_mask(0xFFFFFFFF);

    SIMTStackEntry entry;
    entry.branch_pc = 0;
    entry.reconvergence_pc = 4;
    entry.active_mask = 0xFFFFFFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 4;
    warp->get_simt_stack().push(entry);

    REQUIRE(warp->get_simt_stack().depth() == 1);

    for (int i = 0; i < 32; i++) {
        warp->set_thread_pc(i, 4);
    }
    warp->check_reconvergence();

    REQUIRE(warp->get_simt_stack().empty() == true);
    REQUIRE(warp->get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("ShortestFirst mode with divergent paths", "[shortest_first][divergent]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    SMContext sm(4, 128, 65536, 0);
    sm.set_divergence_execution_mode(ptxsim::DivergenceExecutionMode::ShortestFirst);

    std::vector<StatementContext> statements;
    for (int i = 0; i < 16; i++) {
        statements.push_back(make_nop_stmt());
    }
    statements.push_back(make_mov_stmt());

    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    for (int i = 0; i < 32; i++) {
        warp->get_warp_state().threads[i].pc = 0;
        warp->get_warp_state().threads[i].next_pc = 0;
        warp->get_warp_state().threads[i].is_active = true;
        warp->get_warp_state().threads[i].is_exited = false;
    }

    SIMTStackEntry entry;
    entry.branch_pc = 0;
    entry.reconvergence_pc = 16;
    entry.active_mask = 0xFFFFFFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 16;
    warp->get_simt_stack().push(entry);

    warp->set_exec_mask(0xFFFFFFFF);

    REQUIRE(warp->get_simt_stack().depth() == 1);
}
