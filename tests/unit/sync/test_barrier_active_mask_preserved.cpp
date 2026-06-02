#include "catch_amalgamated.hpp"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "memory/resource_manager.h"
#include <map>
#include <memory>

TEST_CASE("cta_barrier_complete_preserves_exec_mask", "[barrier][active_mask][bug]") {
    ResourceManager::instance().initialize(1, 8192);

    SMContext sm(4, 128, 4096, 0);

    std::unique_ptr<CTAContext> block = std::make_unique<CTAContext>();
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};

    std::vector<StatementContext> statements;
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    block->init(gridDim, blockDim, blockIdx, statements, &name2Sym, label2pc);
    block->sharedMemBytes = 1024;

    bool success = sm.add_block(std::move(block));
    REQUIRE(success == true);

    WarpContext* warp = sm.get_warp(0);
    REQUIRE(warp != nullptr);

    warp->set_exec_mask(0x0000FFFF);
    warp->set_active_mask(0x0000FFFF);

    REQUIRE(warp->get_exec_mask() == 0x0000FFFF);
    REQUIRE(warp->get_active_mask() == 0x0000FFFF);

    for (int i = 0; i < 32; i++) {
        ThreadContext* t = warp->get_thread(i);
        REQUIRE(t != nullptr);
        t->set_state(RUN);
        sm.synchronize_barrier(0, t);
    }

    CHECK(warp->get_exec_mask() == 0x0000FFFF);
}

TEST_CASE("cta_barrier_complete_preserves_active_mask", "[barrier][active_mask][bug]") {
    ResourceManager::instance().initialize(1, 8192);

    SMContext sm(4, 128, 4096, 0);

    std::unique_ptr<CTAContext> block = std::make_unique<CTAContext>();
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};

    std::vector<StatementContext> statements;
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    block->init(gridDim, blockDim, blockIdx, statements, &name2Sym, label2pc);
    block->sharedMemBytes = 1024;

    bool success = sm.add_block(std::move(block));
    REQUIRE(success == true);

    WarpContext* warp = sm.get_warp(0);
    REQUIRE(warp != nullptr);

    warp->set_exec_mask(0x0000FFFF);
    warp->set_active_mask(0x0000FFFF);

    REQUIRE(warp->get_exec_mask() == 0x0000FFFF);
    REQUIRE(warp->get_active_mask() == 0x0000FFFF);

    for (int i = 0; i < 32; i++) {
        ThreadContext* t = warp->get_thread(i);
        REQUIRE(t != nullptr);
        t->set_state(RUN);
        sm.synchronize_barrier(0, t);
    }

    CHECK(warp->get_active_mask() == 0x0000FFFF);
}

TEST_CASE("cta_barrier_already_completed_preserves_exec_mask", "[barrier][active_mask][bug]") {
    ResourceManager::instance().initialize(1, 8192);

    SMContext sm(4, 128, 4096, 0);

    std::unique_ptr<CTAContext> block = std::make_unique<CTAContext>();
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};

    std::vector<StatementContext> statements;
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    block->init(gridDim, blockDim, blockIdx, statements, &name2Sym, label2pc);
    block->sharedMemBytes = 1024;

    bool success = sm.add_block(std::move(block));
    REQUIRE(success == true);

    WarpContext* warp = sm.get_warp(0);
    REQUIRE(warp != nullptr);

    warp->set_exec_mask(0x0000FFFF);
    warp->set_active_mask(0x0000FFFF);

    REQUIRE(warp->get_exec_mask() == 0x0000FFFF);
    REQUIRE(warp->get_active_mask() == 0x0000FFFF);

    for (int i = 0; i < 32; i++) {
        ThreadContext* t = warp->get_thread(i);
        REQUIRE(t != nullptr);
        t->set_state(RUN);
        sm.synchronize_barrier(0, t);
    }

    REQUIRE(warp->get_exec_mask() == 0x0000FFFF);

    ThreadContext* t0 = warp->get_thread(0);
    REQUIRE(t0 != nullptr);
    t0->set_state(RUN);

    sm.synchronize_barrier(0, t0);

    CHECK(warp->get_exec_mask() == 0x0000FFFF);
}
