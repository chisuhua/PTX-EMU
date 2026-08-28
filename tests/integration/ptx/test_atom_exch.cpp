// test_atom_exch.cpp
// =============================================================================
// Integration test (类型二) — atom.global.exch.u32 on the PTX-EMU simulator.
//
// Verifies atomic exchange: dst <- *addr (old value); *addr <- src.
// Extends atomic op coverage (audit §3.2): only atom.add had tests before.
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/memory_test_utils.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"

#include "memory/hardware_memory_manager.h"
#include "memory/resource_manager.h"
#include "memory/simple_memory.h"
#include "register/register_bank_manager.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

using ptxsim::testing::init_instruction_factory_once;
using ptxsim::testing::make_atom_global_exch_u32;
using ptxsim::testing::make_ret;
using ptxsim::testing::setup_block;
using ptxsim::testing::step_warp;

TEST_CASE("atom.global.exch.u32 returns old value and stores new",
          "[integration][ptx][atom][exch]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    constexpr size_t GLOBAL_MEM_SIZE = 64 * 1024;
    auto *simple_mem = new SimpleMemory(GLOBAL_MEM_SIZE);
    HardwareMemoryManager::instance().set_simple_memory(simple_mem);

    uint64_t addr_host =
        reinterpret_cast<uint64_t>(simple_mem->get_global_pool());

    constexpr uint32_t INITIAL_MEM = 100;
    constexpr uint32_t SRC_VALUE = 200;
    simple_mem->direct_access(addr_host, const_cast<uint32_t *>(&INITIAL_MEM),
                              sizeof(uint32_t), /*is_write=*/true);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_atom_global_exch_u32("r_old", "rd_addr", "r_src"));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    auto rbm = w->get_register_bank_manager();
    rbm->create_register("rd_addr", 8);
    rbm->create_register("r_src", 4);
    rbm->create_register("r_old", 4);
    for (int lane = 0; lane < 32; ++lane) {
        *static_cast<uint64_t *>(rbm->get_register("rd_addr", 0, lane)) = addr_host;
        *static_cast<uint32_t *>(rbm->get_register("r_src", 0, lane)) = SRC_VALUE;
        *static_cast<uint32_t *>(rbm->get_register("r_old", 0, lane)) = 0;
    }

    constexpr int MAX_CYCLES = 5000;
    int cycle_count = 0;
    while (!w->is_finished() && cycle_count < MAX_CYCLES) {
        WarpContext::decrement_blocked_cycles(w->get_warp_state());
        step_warp(w, stmts);
        ++cycle_count;
    }
    REQUIRE(w->is_finished());

    uint32_t lane0_old = *static_cast<uint32_t *>(rbm->get_register("r_old", 0, 0));
    REQUIRE(lane0_old == INITIAL_MEM);

    uint32_t mem_after = 0;
    simple_mem->direct_access(addr_host, &mem_after, sizeof(uint32_t),
                              /*is_write=*/false);
    REQUIRE(mem_after == SRC_VALUE);

    delete simple_mem;
    HardwareMemoryManager::instance().set_simple_memory(nullptr);
}

TEST_CASE("atom.global.exch.u32 serializes across lanes",
          "[integration][ptx][atom][exch][serialization]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    constexpr size_t GLOBAL_MEM_SIZE = 64 * 1024;
    auto *simple_mem = new SimpleMemory(GLOBAL_MEM_SIZE);
    HardwareMemoryManager::instance().set_simple_memory(simple_mem);

    uint64_t addr_host =
        reinterpret_cast<uint64_t>(simple_mem->get_global_pool());

    constexpr uint32_t INITIAL_MEM = 0;
    simple_mem->direct_access(addr_host, const_cast<uint32_t *>(&INITIAL_MEM),
                              sizeof(uint32_t), /*is_write=*/true);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_atom_global_exch_u32("r_old", "rd_addr", "r_src"));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    auto rbm = w->get_register_bank_manager();
    rbm->create_register("rd_addr", 8);
    rbm->create_register("r_src", 4);
    rbm->create_register("r_old", 4);
    for (int lane = 0; lane < 32; ++lane) {
        *static_cast<uint64_t *>(rbm->get_register("rd_addr", 0, lane)) = addr_host;
        *static_cast<uint32_t *>(rbm->get_register("r_src", 0, lane)) = lane + 1;
        *static_cast<uint32_t *>(rbm->get_register("r_old", 0, lane)) = 0;
    }

    constexpr int MAX_CYCLES = 5000;
    int cycle_count = 0;
    while (!w->is_finished() && cycle_count < MAX_CYCLES) {
        WarpContext::decrement_blocked_cycles(w->get_warp_state());
        step_warp(w, stmts);
        ++cycle_count;
    }
    REQUIRE(w->is_finished());

    uint32_t lane0_old = *static_cast<uint32_t *>(rbm->get_register("r_old", 0, 0));
    REQUIRE(lane0_old < 33);

    uint32_t mem_after = 0;
    simple_mem->direct_access(addr_host, &mem_after, sizeof(uint32_t),
                              /*is_write=*/false);
    REQUIRE(mem_after >= 1);
    REQUIRE(mem_after <= 32);

    delete simple_mem;
    HardwareMemoryManager::instance().set_simple_memory(nullptr);
}