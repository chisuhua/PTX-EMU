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

TEST_CASE("integrated_barrier_wbar_arrive_and_complete", "[barrier][integrated][execute_warp]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 2));
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    step_warp(warp, statements);
    step_warp(warp, statements);

    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
        CHECK(!warp->get_warp_state().threads[i].is_blocked);
    }

    step_warp(warp, statements);

    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 3);
    }
}

TEST_CASE("integrated_barrier_partial_participants", "[barrier][partial][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0x0000FFFF, 2));
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    for (int i = 16; i < 32; i++) {
        warp->get_warp_state().threads[i].is_active = false;
    }
    warp->set_active_mask(0x0000FFFF);
    warp->set_exec_mask(0x0000FFFF);

    step_warp(warp, statements);
    step_warp(warp, statements);

    for (int i = 0; i < 16; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
        CHECK(!warp->get_warp_state().threads[i].is_blocked);
    }
    for (int i = 16; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 0);
    }
}

TEST_CASE("integrated_barrier_reset_and_reuse", "[barrier][lifecycle][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 2));
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 4));
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    step_warp(warp, statements);
    step_warp(warp, statements);

    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 2);
    }

    step_warp(warp, statements);
    step_warp(warp, statements);

    for (int i = 0; i < 32; i++) {
        CHECK(warp->get_thread(i)->get_pc() == 4);
    }
}
}  // anonymous namespace
