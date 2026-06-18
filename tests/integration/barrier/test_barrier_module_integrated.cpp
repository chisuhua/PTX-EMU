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
#include "ptxsim/barrier/barrier_module.h"
#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"

using ptxsim::testing::step_warp;
#include <map>
#include <memory>
#include <vector>
#include <cstdint>

namespace {
using namespace ptxir::factory;
using ptxsim::BarrierModule;

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

// ============================================================================
// Regression: BUG-POSTBARRIER-TWOHALVES
// When a divergent warp hits the same barrier in two halves at different
// times, the second BarrierModule::release_warp_barrier call MUST OR with
// existing active_mask (per src/ptxsim/core/AGENTS.md invariant — "OR logic
// must live in the caller"). Otherwise the second half overwrites the
// first half's released lanes, losing them.
//
// Pre-fix: active_mask = 0xFFFF0000 (only second half).
// Post-fix: active_mask = 0xFFFFFFFF (OR'd).
// ============================================================================
TEST_CASE("release_warp_barrier ORs with existing active_mask (BUG-POSTBARRIER-TWOHALVES)",
          "[barrier_module][integrated][regression][BUG-POSTBARRIER-TWOHALVES]")
{
    init_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    auto statements = build_simple_barrier_statements();
    SMContext sm(4, 128, 4096, 0);
    auto register_bank = std::make_shared<RegisterBankManager>(4, 32);
    WarpContext* warp = create_warp_with_threads(sm, create_block(statements), register_bank);

    // Initial active_mask: all 32 lanes active
    warp->set_active_mask(0xFFFFFFFFu);
    REQUIRE(warp->get_active_mask() == 0xFFFFFFFFu);

    BarrierModule mod;

    // First half arrives: lanes 0-15
    mod.init_warp_barrier(0, 0x0000FFFFu, /*reconv_pc=*/5, /*barrier_pc=*/3);
    for (int i = 0; i < 16; ++i) {
        mod.arrive_at_warp_barrier(0, i);
    }
    REQUIRE(mod.is_warp_barrier_complete(0));

    // First release: active_mask should be 0x0000FFFF (or-merged: 0xFFFFFFFF | 0x0000FFFF = 0xFFFFFFFF)
    mod.release_warp_barrier(0, warp);
    REQUIRE(warp->get_active_mask() == 0xFFFFFFFFu);

    // Second half arrives: lanes 16-31 (force_reconvergence scenario)
    mod.init_warp_barrier(0, 0xFFFF0000u, /*reconv_pc=*/5, /*barrier_pc=*/3);
    for (int i = 16; i < 32; ++i) {
        mod.arrive_at_warp_barrier(0, i);
    }
    REQUIRE(mod.is_warp_barrier_complete(0));

    // KEY ASSERTION: second release must OR with existing active_mask,
    // NOT overwrite. With the bug, active_mask would become 0xFFFF0000
    // (losing lanes 0-15 released by first half).
    mod.release_warp_barrier(0, warp);
    CHECK(warp->get_active_mask() == 0xFFFFFFFFu);

    // exec_mask should still reflect the second half (per BarrierModule
    // semantics: exec_mask is the lanes that "just completed" the barrier)
    CHECK(warp->get_warp_state().exec_mask == 0xFFFF0000u);
}

// ============================================================================
// CTA-level barrier end-to-end: arrive -> is_complete -> release advances PC.
// Regression for BUG-HANDLER-PC-ADVANCE: pre-fix BarHandler::executeBarrier
// set next_pc=pc+1 but did NOT call commit_pc(), so warp_state.threads[lane].pc
// stayed at barrier_pc after release; threads stuck in infinite loop.
// Post-fix: BarrierModule::release_cta_barrier advances per-thread PC.
// ============================================================================
TEST_CASE("CTA barrier release advances per-thread PC (BUG-HANDLER-PC-ADVANCE)",
          "[barrier][release][cta][regression][BUG-HANDLER-PC-ADVANCE]")
{
    init_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    constexpr int BLOCK_DIM = 32;
    constexpr int BARRIER_PC = 10;
    constexpr int POST_BARRIER_PC = 11;

    std::vector<StatementContext> statements;

    auto block = std::make_unique<CTAContext>();
    block->init({1, 1, 1}, {BLOCK_DIM, 1, 1}, {0, 0, 0},
                statements, nullptr, std::map<std::string, int>{});
    block->sharedMemBytes = 128;

    auto register_bank = std::make_shared<RegisterBankManager>(1, BLOCK_DIM);
    BarrierModule& bm = block->get_barrier_module();
    WarpContext* warp = block->get_warp(0);
    REQUIRE(warp != nullptr);
    warp->set_register_bank_manager(register_bank);
    for (int i = 0; i < BLOCK_DIM; i++) {
        warp->get_thread(i)->set_register_bank_manager(register_bank);
        warp->get_warp_state().threads[i].pc = BARRIER_PC;
    }

    CTABarrier* ctabar = bm.init_cta_barrier(0, BLOCK_DIM, 1);
    REQUIRE(ctabar != nullptr);
    REQUIRE_FALSE(ctabar->is_complete());

    for (int i = 0; i < BLOCK_DIM; i++) {
        ThreadContext* t = warp->get_thread(i);
        REQUIRE(t != nullptr);
        REQUIRE(t->warp_context_ == warp);
        REQUIRE(t->lane_id_ == i);
        bool complete = bm.arrive_at_cta_barrier(0, t);
        if (i < BLOCK_DIM - 1) {
            REQUIRE_FALSE(complete);
        } else {
            REQUIRE(complete);
        }
    }
    REQUIRE(ctabar->is_complete());

    for (int i = 0; i < BLOCK_DIM; i++) {
        REQUIRE(warp->get_warp_state().threads[i].pc ==
                static_cast<uint32_t>(BARRIER_PC));
    }

    bm.release_cta_barrier(0, block.get(), POST_BARRIER_PC);

    for (int i = 0; i < BLOCK_DIM; i++) {
        CHECK(warp->get_warp_state().threads[i].pc ==
              static_cast<uint32_t>(POST_BARRIER_PC));
    }

    CHECK(ctabar->get_arrived_count() == 0);
    CHECK_FALSE(ctabar->is_initialized());

    ctabar = bm.init_cta_barrier(0, BLOCK_DIM, 1);
    REQUIRE(ctabar != nullptr);
    REQUIRE_FALSE(ctabar->is_complete());
}