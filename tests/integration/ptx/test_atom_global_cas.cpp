// test_atom_global_cas.cpp
// =============================================================================
// Integration test (类型二) — atom.global.cas.u32 on the PTX-EMU simulator.
//
// Verifies atomic Compare-And-Swap: dst <- *addr (old value); if *addr == cmp,
// *addr <- val (swap); otherwise *addr unchanged. Implements the OpenSpec
// change `implement-atomic-cas-and-true-atomicity` Phase 1 (CAS handler).
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
using ptxsim::testing::make_atom_global_cas_u32;
using ptxsim::testing::make_ret;
using ptxsim::testing::setup_block;
using ptxsim::testing::step_warp;

// CAS 语义验证: cmp == old → 写入 val, dst = old, mem = val
TEST_CASE("atom.global.cas.u32 writes val when cmp matches and returns old",
          "[integration][ptx][atom][cas][match]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    constexpr size_t GLOBAL_MEM_SIZE = 64 * 1024;
    auto *simple_mem = new SimpleMemory(GLOBAL_MEM_SIZE);
    HardwareMemoryManager::instance().set_simple_memory(simple_mem);

    uint64_t addr_host =
        reinterpret_cast<uint64_t>(simple_mem->get_global_pool());

    constexpr uint32_t INITIAL_MEM = 100;
    constexpr uint32_t CMP_VALUE = 100;  // matches INITIAL_MEM
    constexpr uint32_t VAL_VALUE = 999;
    simple_mem->direct_access(addr_host, const_cast<uint32_t *>(&INITIAL_MEM),
                              sizeof(uint32_t), /*is_write=*/true);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_atom_global_cas_u32("r_old", "rd_addr", "r_cmp",
                                              "r_val"));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    auto rbm = w->get_register_bank_manager();
    rbm->create_register("rd_addr", 8);
    rbm->create_register("r_cmp", 4);
    rbm->create_register("r_val", 4);
    rbm->create_register("r_old", 4);
    for (int lane = 0; lane < 32; ++lane) {
        *static_cast<uint64_t *>(rbm->get_register("rd_addr", 0, lane)) =
            addr_host;
        *static_cast<uint32_t *>(rbm->get_register("r_cmp", 0, lane)) =
            CMP_VALUE;
        *static_cast<uint32_t *>(rbm->get_register("r_val", 0, lane)) =
            VAL_VALUE;
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

    // Single-warp serialized dispatcher (sm_context.cpp) executes lanes in
    // some order; with CMP == INITIAL == INITIAL for every lane, the first
    // lane to load sees INITIAL and writes VAL; every subsequent lane loads
    // VAL, compares against CMP == INITIAL, and skips the store. The dst
    // register therefore contains the value loaded by that lane's read,
    // which is either INITIAL_MEM (first/lucky lane) or VAL_VALUE (any
    // later lane that loaded after the first write).
    bool saw_initial_old = false;
    bool saw_val_old = false;
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t lane_old =
            *static_cast<uint32_t *>(rbm->get_register("r_old", 0, lane));
        REQUIRE((lane_old == INITIAL_MEM || lane_old == VAL_VALUE));
        if (lane_old == INITIAL_MEM) saw_initial_old = true;
        if (lane_old == VAL_VALUE) saw_val_old = true;
    }
    // At least one lane must have observed the original value (proves the
    // first-issued read precedes the write) and at least one must have
    // observed val (proves a subsequent read picked up the write).
    REQUIRE(saw_initial_old);
    REQUIRE(saw_val_old);

    // Memory ends as VAL because the first lane's CAS succeeded.
    uint32_t mem_after = 0;
    simple_mem->direct_access(addr_host, &mem_after, sizeof(uint32_t),
                              /*is_write=*/false);
    REQUIRE(mem_after == VAL_VALUE);

    delete simple_mem;
    HardwareMemoryManager::instance().set_simple_memory(nullptr);
}

// CAS 语义验证: cmp != old → 不写入 val, dst = old, mem = old unchanged
TEST_CASE("atom.global.cas.u32 leaves mem unchanged when cmp mismatches",
          "[integration][ptx][atom][cas][mismatch]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    constexpr size_t GLOBAL_MEM_SIZE = 64 * 1024;
    auto *simple_mem = new SimpleMemory(GLOBAL_MEM_SIZE);
    HardwareMemoryManager::instance().set_simple_memory(simple_mem);

    uint64_t addr_host =
        reinterpret_cast<uint64_t>(simple_mem->get_global_pool());

    constexpr uint32_t INITIAL_MEM = 100;
    constexpr uint32_t CMP_VALUE = 5;  // does NOT match INITIAL_MEM
    constexpr uint32_t VAL_VALUE = 999;
    simple_mem->direct_access(addr_host, const_cast<uint32_t *>(&INITIAL_MEM),
                              sizeof(uint32_t), /*is_write=*/true);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_atom_global_cas_u32("r_old", "rd_addr", "r_cmp",
                                              "r_val"));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    auto rbm = w->get_register_bank_manager();
    rbm->create_register("rd_addr", 8);
    rbm->create_register("r_cmp", 4);
    rbm->create_register("r_val", 4);
    rbm->create_register("r_old", 4);
    for (int lane = 0; lane < 32; ++lane) {
        *static_cast<uint64_t *>(rbm->get_register("rd_addr", 0, lane)) =
            addr_host;
        *static_cast<uint32_t *>(rbm->get_register("r_cmp", 0, lane)) =
            CMP_VALUE;
        *static_cast<uint32_t *>(rbm->get_register("r_val", 0, lane)) =
            VAL_VALUE;
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

    // dst 仍必须等于 old
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t lane_old =
            *static_cast<uint32_t *>(rbm->get_register("r_old", 0, lane));
        REQUIRE(lane_old == INITIAL_MEM);
    }

    // 内存必须保持 INITIAL_MEM 不变 (因为没有任何 lane 的 cmp 匹配)
    uint32_t mem_after = 0;
    simple_mem->direct_access(addr_host, &mem_after, sizeof(uint32_t),
                              /*is_write=*/false);
    REQUIRE(mem_after == INITIAL_MEM);

    delete simple_mem;
    HardwareMemoryManager::instance().set_simple_memory(nullptr);
}

// CAS 混合场景: 部分 lane cmp 匹配,部分不匹配 → winner-takes-all
TEST_CASE("atom.global.cas.u32 winner-takes-all on mixed cmp",
          "[integration][ptx][atom][cas][mixed]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    constexpr size_t GLOBAL_MEM_SIZE = 64 * 1024;
    auto *simple_mem = new SimpleMemory(GLOBAL_MEM_SIZE);
    HardwareMemoryManager::instance().set_simple_memory(simple_mem);

    uint64_t addr_host =
        reinterpret_cast<uint64_t>(simple_mem->get_global_pool());

    constexpr uint32_t INITIAL_MEM = 100;
    constexpr uint32_t VAL_VALUE = 999;
    simple_mem->direct_access(addr_host, const_cast<uint32_t *>(&INITIAL_MEM),
                              sizeof(uint32_t), /*is_write=*/true);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_atom_global_cas_u32("r_old", "rd_addr", "r_cmp",
                                              "r_val"));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    auto rbm = w->get_register_bank_manager();
    rbm->create_register("rd_addr", 8);
    rbm->create_register("r_cmp", 4);
    rbm->create_register("r_val", 4);
    rbm->create_register("r_old", 4);
    for (int lane = 0; lane < 32; ++lane) {
        *static_cast<uint64_t *>(rbm->get_register("rd_addr", 0, lane)) =
            addr_host;
        *static_cast<uint32_t *>(rbm->get_register("r_val", 0, lane)) =
            VAL_VALUE;
        // 前 16 lanes: cmp == old (winner), 后 16 lanes: cmp != old (loser)
        *static_cast<uint32_t *>(rbm->get_register("r_cmp", 0, lane)) =
            (lane < 16) ? INITIAL_MEM : (INITIAL_MEM + 1);
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

    // Mixed cmp: first 16 lanes' cmp matches INITIAL; later 16 lanes' cmp
    // does not. Under serialized single-warp dispatch, the first lane to
    // execute among lanes 0..15 will succeed and write VAL; subsequent
    // loads see VAL and either match (if from [0,15] group) or skip (if
    // from [16,31] group). Per-lane dst is therefore whatever each lane's
    // load returned: INITIAL_MEM (lanes that read before any write) or
    // VAL_VALUE (lanes that read after the first winner). At least one lane
    // in the [0,15] group must observe INITIAL_MEM for a CAS to have run
    // against the original value.
    bool saw_initial_old = false;
    bool saw_val_old = false;
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t lane_old =
            *static_cast<uint32_t *>(rbm->get_register("r_old", 0, lane));
        REQUIRE((lane_old == INITIAL_MEM || lane_old == VAL_VALUE));
        if (lane_old == INITIAL_MEM) saw_initial_old = true;
        if (lane_old == VAL_VALUE) saw_val_old = true;
    }
    REQUIRE(saw_initial_old);
    REQUIRE(saw_val_old);

    // Memory is VAL because at least one lane from [0,15] executed a
    // successful CAS before any other lane.
    uint32_t mem_after = 0;
    simple_mem->direct_access(addr_host, &mem_after, sizeof(uint32_t),
                              /*is_write=*/false);
    REQUIRE(mem_after == VAL_VALUE);

    delete simple_mem;
    HardwareMemoryManager::instance().set_simple_memory(nullptr);
}
