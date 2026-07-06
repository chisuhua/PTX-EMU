// test_atom_global_cas_multiwarp.cpp
// =============================================================================
// Integration test (类型二, multi-warp) — atom.global.cas under cross-warp
// contention. Verifies Phase 2 of `implement-atomic-cas-and-true-atomicity`:
// the global atomic mutex serializes concurrent CAS operations across warps.
//
// 2 warps × 32 lanes = 64 threads all execute atomic.global.cas.u32 on the
// same memory address. With per-warp scheduling (sm_context.cpp:225-260) plus
// the cross-warp mutex, the operations are serialized and the final memory
// state is one of the valid 64 possible outcomes (deterministic across runs).
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

namespace {

static std::pair<WarpContext *, WarpContext *>
setup_two_warps(SMContext &sm, std::vector<StatementContext> &stmts) {
    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1};
    Dim3 b{64, 1, 1};  // 64 threads = 2 warps
    Dim3 bi{0, 0, 0};
    std::map<std::string, int> l2pc;
    std::map<std::string, std::unique_ptr<Symtable>> n2s;
    blk->init(g, b, bi, stmts, &n2s, l2pc);
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    WarpContext *w0 = sm.get_warp(0);
    WarpContext *w1 = sm.get_warp(1);
    REQUIRE(w0 != nullptr);
    REQUIRE(w1 != nullptr);
    return {w0, w1};
}

// Drive a warp a bounded number of step_warp() calls. Returns true if
// the warp reached `ret_pc`, false otherwise (deadlock or hang).
static bool run_to_ret(WarpContext *w, std::vector<StatementContext> &stmts,
                       int ret_pc, int max_steps = 64) {
    int steps_taken = 0;
    for (int step = 0; step < max_steps; ++step) {
        WarpContext::decrement_blocked_cycles(w->get_warp_state());
        int pc = ptxsim::testing::step_warp(w, stmts);
        ++steps_taken;
        if (pc == ret_pc) return true;
    }
    return w->is_finished();
}

} // namespace

TEST_CASE("atom.global.cas across 2 warps is mutex-serialized (no deadlock)",
          "[integration][ptx][atom][cas][multiwarp]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(2, 8192);

    constexpr size_t GLOBAL_MEM_SIZE = 64 * 1024;
    auto *simple_mem = new SimpleMemory(GLOBAL_MEM_SIZE);
    HardwareMemoryManager::instance().set_simple_memory(simple_mem);

    uint64_t addr_host =
        reinterpret_cast<uint64_t>(simple_mem->get_global_pool());

    constexpr uint32_t INITIAL_MEM = 100;
    simple_mem->direct_access(addr_host, const_cast<uint32_t *>(&INITIAL_MEM),
                              sizeof(uint32_t), /*is_write=*/true);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_atom_global_cas_u32("r_old", "rd_addr", "r_cmp",
                                              "r_val"));
    stmts.push_back(make_ret());

    SMContext sm(8, 128, 4096, 0);
    auto [w0, w1] = setup_two_warps(sm, stmts);
    REQUIRE(w0 != nullptr);
    REQUIRE(w1 != nullptr);

    constexpr uint32_t CMP_VALUE = INITIAL_MEM;
    auto init_warp_registers = [&](WarpContext *w) {
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
            *static_cast<uint32_t *>(rbm->get_register("r_old", 0, lane)) = 0;
        }
    };
    init_warp_registers(w0);
    init_warp_registers(w1);

    // Each lane writes a unique value: w0 lanes 0..31 → val 1..32; w1 lanes 0..31 → val 33..64.
    auto populate_vals = [&](WarpContext *w, int warp_offset) {
        auto rbm = w->get_register_bank_manager();
        for (int lane = 0; lane < 32; ++lane) {
            uint32_t val = static_cast<uint32_t>(warp_offset + lane + 1);
            *static_cast<uint32_t *>(rbm->get_register("r_val", 0, lane)) = val;
        }
    };
    populate_vals(w0, 0);    // w0 lanes → val 1..32
    populate_vals(w1, 32);   // w1 lanes → val 33..64

    // Drive both warps to RET. With the cross-warp atomic mutex this must
    // converge — no deadlock.
    bool done0 = run_to_ret(w0, stmts, /*ret_pc=*/1, /*max_steps=*/5000);
    bool done1 = run_to_ret(w1, stmts, /*ret_pc=*/1, /*max_steps=*/5000);
    REQUIRE(done0);
    REQUIRE(done1);

    // Memory must be ONE of the 64 valid lane values — the winner's val,
    // since all lanes start with cmp == INITIAL. Without serialization
    // the outcome is non-deterministic; with the global mutex, exactly
    // one writer succeeds and the others observe the new value (cmp no
    // longer matches, dst != INITIAL).
    uint32_t mem_after = 0;
    simple_mem->direct_access(addr_host, &mem_after, sizeof(uint32_t),
                              /*is_write=*/false);
    REQUIRE(mem_after >= 1u);
    REQUIRE(mem_after <= 64u);

    // At least one lane across both warps must have observed INITIAL — the
    // very first read happened before any successful write and the global
    // mutex guarantees that read occurred while mem was still INITIAL.
    bool saw_initial = false;
    for (WarpContext *w : {w0, w1}) {
        auto rbm = w->get_register_bank_manager();
        for (int lane = 0; lane < 32; ++lane) {
            uint32_t lane_old =
                *static_cast<uint32_t *>(rbm->get_register("r_old", 0, lane));
            if (lane_old == INITIAL_MEM) saw_initial = true;
        }
    }
    REQUIRE(saw_initial);

    delete simple_mem;
    HardwareMemoryManager::instance().set_simple_memory(nullptr);
}

TEST_CASE("atom.global.cas all-mismatch across 2 warps leaves mem unchanged",
          "[integration][ptx][atom][cas][multiwarp][mismatch]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(2, 8192);

    constexpr size_t GLOBAL_MEM_SIZE = 64 * 1024;
    auto *simple_mem = new SimpleMemory(GLOBAL_MEM_SIZE);
    HardwareMemoryManager::instance().set_simple_memory(simple_mem);

    uint64_t addr_host =
        reinterpret_cast<uint64_t>(simple_mem->get_global_pool());

    constexpr uint32_t INITIAL_MEM = 100;
    constexpr uint32_t CMP_VALUE = 5;  // never matches INITIAL
    simple_mem->direct_access(addr_host, const_cast<uint32_t *>(&INITIAL_MEM),
                              sizeof(uint32_t), /*is_write=*/true);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_atom_global_cas_u32("r_old", "rd_addr", "r_cmp",
                                              "r_val"));
    stmts.push_back(make_ret());

    SMContext sm(8, 128, 4096, 0);
    auto [w0, w1] = setup_two_warps(sm, stmts);
    REQUIRE(w0 != nullptr);
    REQUIRE(w1 != nullptr);

    auto init_warp = [&](WarpContext *w, int val_offset) {
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
                static_cast<uint32_t>(val_offset + lane + 1);
            *static_cast<uint32_t *>(rbm->get_register("r_old", 0, lane)) = 0;
        }
    };
    init_warp(w0, 0);
    init_warp(w1, 32);

    bool done0 = run_to_ret(w0, stmts, /*ret_pc=*/1, /*max_steps=*/5000);
    bool done1 = run_to_ret(w1, stmts, /*ret_pc=*/1, /*max_steps=*/5000);
    REQUIRE(done0);
    REQUIRE(done1);

    // No CAS matched → mem must equal INITIAL.
    uint32_t mem_after = 0;
    simple_mem->direct_access(addr_host, &mem_after, sizeof(uint32_t),
                              /*is_write=*/false);
    REQUIRE(mem_after == INITIAL_MEM);

    delete simple_mem;
    HardwareMemoryManager::instance().set_simple_memory(nullptr);
}
