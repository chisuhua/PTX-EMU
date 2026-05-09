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
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;
    block->init(gridDim, blockDim, blockIdx, statements, &name2Sym, label2pc);
    return block;
}

TEST_CASE("integrated_pc_after_barrier_commit_flow", "[pc][integrated][execute_warp]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 2));
    statements.push_back(make_mov_stmt());
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    warp->execute_warp_instruction(statements[0], 0);
    warp->execute_warp_instruction(statements[1], 1);

    CHECK(warp->get_wbar(0).is_complete() == true);
    CHECK(warp->is_warp_ready_to_fetch() == true);

    warp->execute_warp_instruction(statements[2], 2);

    for (int i = 0; i < 32; i++) {
        if (warp->get_warp_state().threads[i].pc != warp->get_warp_state().threads[i].next_pc) {
            CHECK(warp->is_warp_ready_to_fetch() == false);
        }
    }

    warp->execute_warp_instruction(statements[3], 3);
    CHECK(warp->is_warp_ready_to_fetch() == true);
}

TEST_CASE("integrated_pc_divergent_commit", "[pc][divergence][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(make_mov_stmt());
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    warp->get_warp_state().exec_mask = 0xFFFF0000;

    for (int i = 0; i < 16; i++) {
        warp->get_warp_state().threads[i].next_pc = 20;
    }
    for (int i = 16; i < 32; i++) {
        warp->get_warp_state().threads[i].next_pc = 10;
    }

    for (int i = 0; i < 32; i++) {
        warp->get_thread(i)->commit_pc();
    }

    CHECK(warp->get_warp_state().threads[0].pc == 20);
    CHECK(warp->get_warp_state().threads[16].pc == 10);
    CHECK(warp->is_warp_ready_to_fetch() == true);
}

TEST_CASE("integrated_warp_ready_with_inactive_threads", "[pc][scheduler][integrated]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> statements;
    statements.push_back(make_mov_stmt());
    statements.push_back(makeBarWarpSyncInstr(0x0000FFFF, 2));
    statements.push_back(make_mov_stmt());

    SMContext sm(4, 128, 4096, 0);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements));

    for (int i = 0; i < 16; i++) {
        warp->get_warp_state().threads[i].pc = 10;
        warp->get_warp_state().threads[i].next_pc = 11;
        warp->get_warp_state().threads[i].is_active = false;
    }

    CHECK(warp->is_warp_ready_to_fetch() == true);

    warp->execute_warp_instruction(statements[0], 0);
    warp->execute_warp_instruction(statements[1], 1);

    CHECK(warp->is_warp_ready_to_fetch() == true);
}
}  // anonymous namespace
