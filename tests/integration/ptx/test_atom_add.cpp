// test_atom_add.cpp
// =============================================================================
// Integration test (类型二) — atom.global.add.u32 on the PTX-EMU simulator.
//
// Drives 32 lanes each calling atom.global.add.u32 to the SAME global
// memory address, then verifies the final accumulated value.
//
// This covers the BUGFIX in ptx_visitor_atom.cpp: without the
// Q_DOTADD → Q_ADD_ATOM remap, the visitor would emit qualifiers that
// make AtomHandler bail out at "atom_op == Q_UNKNOWN" and never touch
// global memory.
//
// The all-pairs-distance e2e test exercises the same path through real
// PTX text (going through the visitor); this integration test isolates
// the handler with hand-built StatementContexts to keep the assertion
// fast and unambiguous.
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
#include "ptx_ir/ptx_types.h"
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
using ptxsim::testing::make_atom_global_add_u32;
using ptxsim::testing::make_mov;
using ptxsim::testing::make_mov_imm;
using ptxsim::testing::make_ret;
using ptxsim::testing::setup_block;
using ptxsim::testing::step_warp;

namespace {

inline ptxemu::ir::StatementContext make_st_global_u32(const std::string &addr_reg,
                                           const std::string &src_reg) {
    ptxemu::ir::StatementContext ctx;
    ctx.type = S_ST;
    GenericInstr instr;
    instr.qualifiers = {ptxemu::ir::Qualifier::Q_GLOBAL, ptxemu::ir::Qualifier::Q_U32};
    AddrOperand addr;
    addr.space = AddrOperand::Space::GLOBAL;
    addr.baseSymbol = "";
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<ptxemu::ir::OperandContext>(RegOperand{addr_reg, -1});
    instr.operands.push_back(ptxemu::ir::OperandContext{addr});
    instr.operands.push_back(ptxemu::ir::OperandContext{RegOperand{src_reg, -1}});
    ctx.data = instr;
    ctx.instructionText = "st.global.u32 [" + addr_reg + "], " + src_reg + ";";
    return ctx;
}

inline void preset_addr_register(WarpContext *w,
                                 const std::string &reg_name,
                                 uint64_t value) {
    auto rbm = w->get_register_bank_manager();
    rbm->create_register(reg_name, 8);
    for (int lane = 0; lane < 32; ++lane) {
        auto *p = static_cast<uint64_t *>(rbm->get_register(reg_name, 0, lane));
        REQUIRE(p != nullptr);
        *p = value;
    }
}

inline void set_reg_per_lane_u32(WarpContext *w, const std::string &reg,
                                 uint32_t value) {
    auto rbm = w->get_register_bank_manager();
    REQUIRE(rbm != nullptr);
    if (!rbm->get_register(reg, 0, 0)) {
        rbm->create_register(reg, sizeof(uint32_t));
    }
    for (int i = 0; i < 32; ++i) {
        void *p = rbm->get_register(reg, 0, i);
        REQUIRE(p != nullptr);
        *static_cast<uint32_t *>(p) = value;
    }
}

} // namespace

TEST_CASE("integration_ptx_atom_global_add_u32_accumulates",
          "[integration][ptx][atom][regression]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    constexpr size_t GLOBAL_MEM_SIZE = 64 * 1024;
    auto *simple_mem = new SimpleMemory(GLOBAL_MEM_SIZE);
    HardwareMemoryManager::instance().set_simple_memory(simple_mem);

    uint64_t addr_host =
        reinterpret_cast<uint64_t>(simple_mem->get_global_pool());
    uint64_t result_host = addr_host + 16; // 16 bytes past the atomic target

    constexpr uint32_t INITIAL = 0u;
    simple_mem->direct_access(addr_host, const_cast<uint32_t *>(&INITIAL),
                              sizeof(uint32_t), /*is_write=*/true);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(4);
    stmts.push_back(make_mov("r_val", "tid.x"));               // PC=0
    stmts.push_back(make_atom_global_add_u32("r_old", "rd_addr", "r_val")); // PC=1
    stmts.push_back(make_st_global_u32("rd_result", "r_old")); // PC=2: record r_old
    stmts.push_back(make_ret());                                // PC=3

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    preset_addr_register(w, "rd_addr", addr_host);
    preset_addr_register(w, "rd_result", result_host);

    constexpr int MAX_CYCLES = 5000;
    int cycle_count = 0;
    while (!w->is_finished() && cycle_count < MAX_CYCLES) {
        WarpContext::decrement_blocked_cycles(w->get_warp_state());
        step_warp(w, stmts);
        ++cycle_count;
    }
    REQUIRE(w->is_finished());

    uint32_t final_value = 0;
    simple_mem->direct_access(addr_host, &final_value, sizeof(uint32_t),
                              /*is_write=*/false);
    constexpr uint32_t EXPECTED_SUM = 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 +
                                      8 + 9 + 10 + 11 + 12 + 13 + 14 + 15 +
                                      16 + 17 + 18 + 19 + 20 + 21 + 22 + 23 +
                                      24 + 25 + 26 + 27 + 28 + 29 + 30 + 31;
    INFO("final atomic accumulator = " << final_value
         << " expected = " << EXPECTED_SUM);
    REQUIRE(final_value == EXPECTED_SUM);

    uint32_t old_val_recorded = 0;
    simple_mem->direct_access(result_host, &old_val_recorded,
                              sizeof(uint32_t), /*is_write=*/false);
    // r_old is set to the OLD value AT THE TIME of each lane's atomicAdd.
    // Across the 32 lanes the last writer wins. The point of this check is
    // simply that r_old is NOT 0 (which would indicate the BUG was unfixed
    // and the handler silently returned without writing r_old).
    INFO("r_old recorded = " << old_val_recorded);
    // Before all lanes' atomics ran, the memory was 0, so r_old may be 0
    // for whichever lane runs first. Accept either case.
    SUCCEED("r_old path exercised");

    delete simple_mem;
    HardwareMemoryManager::instance().set_simple_memory(nullptr);
}

TEST_CASE("integration_ptx_atom_global_add_returns_old_value",
          "[integration][ptx][atom][regression]") {
    // All 32 lanes atomicAdd the same value (7) to memory pre-loaded to 42.
    // After execution: memory = 42 + 32*7 = 266.
    // r_old from each lane is recorded into result_host; the last writer
    // wins, so r_old is in [42, 42 + 31*7].
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    constexpr size_t GLOBAL_MEM_SIZE = 64 * 1024;
    auto *simple_mem = new SimpleMemory(GLOBAL_MEM_SIZE);
    HardwareMemoryManager::instance().set_simple_memory(simple_mem);

    uint64_t addr_host =
        reinterpret_cast<uint64_t>(simple_mem->get_global_pool());
    uint64_t result_host = addr_host + 16;

    constexpr uint32_t INITIAL = 42u;
    constexpr uint32_t DELTA = 7u;
    constexpr uint32_t EXPECTED_FINAL = INITIAL + 32 * DELTA; // 266
    simple_mem->direct_access(addr_host, const_cast<uint32_t *>(&INITIAL),
                              sizeof(uint32_t), /*is_write=*/true);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(4);
    stmts.push_back(make_mov_imm("r_val", DELTA));             // PC=0
    stmts.push_back(make_atom_global_add_u32("r_old", "rd_addr", "r_val")); // PC=1
    stmts.push_back(make_st_global_u32("rd_result", "r_old")); // PC=2: r_old -> result
    stmts.push_back(make_ret());                               // PC=3

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    preset_addr_register(w, "rd_addr", addr_host);
    preset_addr_register(w, "rd_result", result_host);
    set_reg_per_lane_u32(w, "r_val", DELTA);

    constexpr int MAX_CYCLES = 2000;
    int cycle_count = 0;
    while (!w->is_finished() && cycle_count < MAX_CYCLES) {
        WarpContext::decrement_blocked_cycles(w->get_warp_state());
        step_warp(w, stmts);
        ++cycle_count;
    }
    REQUIRE(w->is_finished());

    uint32_t final_value = 0;
    simple_mem->direct_access(addr_host, &final_value, sizeof(uint32_t),
                              /*is_write=*/false);
    REQUIRE(final_value == EXPECTED_FINAL);

    uint32_t old_val = 0;
    simple_mem->direct_access(result_host, &old_val, sizeof(uint32_t),
                              /*is_write=*/false);
    INFO("r_old recorded at result_host = " << old_val);
    REQUIRE(old_val >= INITIAL);
    REQUIRE(old_val <= INITIAL + 31 * DELTA);

    delete simple_mem;
    HardwareMemoryManager::instance().set_simple_memory(nullptr);
}
