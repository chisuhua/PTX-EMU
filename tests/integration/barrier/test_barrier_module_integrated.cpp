/**
 * @file test_barrier_module_integrated.cpp
 * @brief BarrierModule 类型二测试：指令序列集成测试
 */

#include "catch_amalgamated.hpp"
#include "ptx_ir/statement_factory.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"

using ptxsim::testing::step_warp;
#include <map>
#include <memory>
#include <vector>
#include <cstdint>

namespace {
using namespace ptxir::factory;

static void init_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

static std::vector<StatementContext> build_simple_barrier_statements() {
    std::vector<StatementContext> stmts;
    stmts.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 1));
    stmts.push_back(makeVoidInstr(S_RET, "ret;"));
    return stmts;
}

static WarpContext* create_warp_with_threads(
    SMContext& sm, std::unique_ptr<CTAContext> block,
    std::shared_ptr<RegisterBankManager> register_bank) {
    block->sharedMemBytes = 128;
    bool ok = sm.add_block(std::move(block));
    REQUIRE(ok == true);
    WarpContext* warp = sm.get_warp(0);
    warp->set_register_bank_manager(register_bank);
    for (int i = 0; i < 32; i++) {
        warp->get_thread(i)->set_register_bank_manager(register_bank);
    }
    return warp;
}

static std::unique_ptr<CTAContext> create_block(
    std::vector<StatementContext>& statements,
    Dim3 gridDim = {1, 1, 1},
    Dim3 blockDim = {32, 1, 1},
    Dim3 blockIdx = {0, 0, 0}) {
    auto block = std::make_unique<CTAContext>();
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;
    block->init(gridDim, blockDim, blockIdx, statements, &name2Sym, label2pc);
    return block;
}

} // anonymous namespace

TEST_CASE("BarrierModule execute barrier instruction", "[barrier_module][integrated][execute_warp]") {
    init_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    auto statements = build_simple_barrier_statements();
    REQUIRE(statements.size() == 2);

    SMContext sm(4, 128, 4096, 0);
    auto register_bank = std::make_shared<RegisterBankManager>(4, 32);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements), register_bank);

    step_warp(warp, statements);
    step_warp(warp, statements);
}

TEST_CASE("BarrierModule multi-warp CTA barrier sharing", "[barrier_module][integrated]") {
    init_factory_once();
    ResourceManager::instance().initialize(2, 8192);

    std::vector<StatementContext> statements = build_simple_barrier_statements();

    SMContext sm(4, 128, 4096, 0);
    auto register_bank = std::make_shared<RegisterBankManager>(4, 32);

    WarpContext* warp0 = create_warp_with_threads(sm, create_block(statements), register_bank);
    REQUIRE(warp0 != nullptr);

    REQUIRE(warp0->get_warp_id() == 0);
}

TEST_CASE("BarrierModule arrive semantics via Wbar", "[barrier_module][integrated]") {
    init_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    auto statements = build_simple_barrier_statements();

    SMContext sm(4, 128, 4096, 0);
    auto register_bank = std::make_shared<RegisterBankManager>(4, 32);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements), register_bank);

    ptxsim::Wbar& wbar = warp->get_wbar(0);

    wbar.init(0x0000FFFF, 1);
    REQUIRE(wbar.count_participants() == 16);
    REQUIRE(!wbar.is_complete());

    for (int i = 0; i < 15; i++) {
        wbar.arrive(i);
        REQUIRE(!wbar.is_complete());
    }

    wbar.arrive(15);
    REQUIRE(wbar.is_complete());
}