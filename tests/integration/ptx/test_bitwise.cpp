// test_bitwise.cpp
// =============================================================================
// Integration test (类型二) — and.b32 / or.b32 / xor.b32 / shl.b32 / not.b32
// on the PTX-EMU simulator (NOT real GPU). Drives SMContext + WarpContext
// + step_warp.
//
// Per TEST_CASE:
//   PC=0:  mov.b32 r1, tid.x    ; r1[lane] = lane_id
//   PC=1:  <bitwise_op>         ; see SECTIONs
//   PC=2:  ret
//
// Expected: r2 == compute(lane_id) for every lane.
//
// Per-lane r1 is set via the RegisterBankManager before step_warp runs.
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"

#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using ptxsim::testing::make_and;
using ptxsim::testing::make_bfe_u32;
using ptxsim::testing::make_mov;
using ptxsim::testing::make_mov_imm;
using ptxsim::testing::make_not;
using ptxsim::testing::make_or;
using ptxsim::testing::make_ret;
using ptxsim::testing::make_shl;
using ptxsim::testing::make_shr;
using ptxsim::testing::make_xor;
using ptxsim::testing::step_warp;

namespace {

void init_instruction_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

void set_reg_per_lane_u32(WarpContext *w, const std::string &reg,
                          std::function<uint32_t(int)> fn) {
    auto rbm = w->get_register_bank_manager();
    REQUIRE(rbm != nullptr);
    if (!rbm->get_register(reg, 0, 0)) {
        rbm->create_register(reg, sizeof(uint32_t));
    }
    for (int i = 0; i < 32; ++i) {
        void *p = rbm->get_register(reg, 0, i);
        REQUIRE(p != nullptr);
        *static_cast<uint32_t *>(p) = fn(i);
    }
}

WarpContext *setup_block(SMContext &sm, std::vector<ptxemu::ir::StatementContext> &stmts) {
    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1};
    Dim3 b{32, 1, 1};
    Dim3 bi{0, 0, 0};
    std::map<std::string, int> l2pc;
    std::map<std::string, std::unique_ptr<Symtable>> n2s;
    blk->init(g, b, bi, stmts, &n2s, l2pc);
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    return sm.get_warp(0);
}

uint32_t get_reg_u32(WarpContext *w, const std::string &reg, int lane) {
    auto rbm = w->get_register_bank_manager();
    void *p = rbm->get_register(reg, 0, lane);
    REQUIRE(p != nullptr);
    return *static_cast<uint32_t *>(p);
}

} // namespace

TEST_CASE("integration_ptx_bitwise_and_b32",
          "[integration][ptx][bitwise][and]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_mov("r1", "tid.x"));    // PC=0
    stmts.push_back(make_and("r2", "r1", "r1")); // PC=1: r2 = r1 & r1 = r1
    stmts.push_back(make_ret());                 // PC=2

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    int ret_pc = -1;
    for (int step = 0; step < 16; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 2) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 2);

    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = get_reg_u32(w, "r2", lane);
        uint32_t expected = static_cast<uint32_t>(lane);
        CHECK(v == expected);
    }
}

TEST_CASE("integration_ptx_bitwise_or_b32", "[integration][ptx][bitwise][or]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_mov("r1", "tid.x"));   // PC=0
    stmts.push_back(make_or("r2", "r1", "r1")); // PC=1: r2 = r1 | r1 = r1
    stmts.push_back(make_ret());                // PC=2

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    int ret_pc = -1;
    for (int step = 0; step < 16; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 2) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 2);

    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = get_reg_u32(w, "r2", lane);
        uint32_t expected = static_cast<uint32_t>(lane);
        CHECK(v == expected);
    }
}

TEST_CASE("integration_ptx_bitwise_xor_b32",
          "[integration][ptx][bitwise][xor]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_mov("r1", "tid.x"));    // PC=0
    stmts.push_back(make_xor("r2", "r1", "r1")); // PC=1: r2 = r1 ^ r1 = 0
    stmts.push_back(make_ret());                 // PC=2

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    int ret_pc = -1;
    for (int step = 0; step < 16; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 2) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 2);

    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = get_reg_u32(w, "r2", lane);
        uint32_t expected = 0u;
        CHECK(v == expected);
    }
}

TEST_CASE("integration_ptx_bitwise_shl_b32",
          "[integration][ptx][bitwise][shl]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_mov("r1", "tid.x"));    // PC=0
    stmts.push_back(make_shl("r2", "r1", "r1")); // PC=1: r2 = r1 << r1
    stmts.push_back(make_ret());                 // PC=2

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    int ret_pc = -1;
    for (int step = 0; step < 16; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 2) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 2);

    // Only lanes 0..4 give meaningful results; for larger shift amounts the
    // shift wraps to 0 (32-bit PTX shl uses lower 5 bits of shift amount).
    for (int lane = 0; lane < 5; ++lane) {
        uint32_t v = get_reg_u32(w, "r2", lane);
        uint32_t expected = static_cast<uint32_t>(lane)
                            << static_cast<uint32_t>(lane);
        CHECK(v == expected);
    }
}

TEST_CASE("integration_ptx_bitwise_not_b32",
          "[integration][ptx][bitwise][not]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_mov("r1", "tid.x")); // PC=0
    stmts.push_back(make_not("r2", "r1"));    // PC=1: r2 = ~r1
    stmts.push_back(make_ret());              // PC=2

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    int ret_pc = -1;
    for (int step = 0; step < 16; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 2) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 2);

    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = get_reg_u32(w, "r2", lane);
        uint32_t expected = ~static_cast<uint32_t>(lane);
        CHECK(v == expected);
    }
}

// ============================================================================
// BFE (Bit Field Extract) regression tests
// ============================================================================
//
// These cover BUGFIX: S_BFE was missing from ptx_op.def, causing all-pairs-
// distance GPUregister to write 0 (bfe.u32 result silently dropped, so the
// per-byte char4 count was always 0, so every atomicAdd added 0).
//
// Test layout per case:
//   PC=0: mov.b32 r1, tid.x  ; r1 = lane_id (per lane)
//   PC=1: mov.u32 r_pos, <pos>
//   PC=2: mov.u32 r_len, <len>
//   PC=3: bfe.u32 r2, r1, r_pos, r_len
//   PC=4: ret
//
// We use registers for pos/len to exercise the full qualifier + operand
// acquisition pipeline that nvcc emits in the all-pairs-distance PTX.

TEST_CASE("integration_ptx_bitwise_bfe_byte_extract",
          "[integration][ptx][bitwise][bfe][regression]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(5);
    stmts.push_back(make_mov("r1", "tid.x"));               // PC=0
    stmts.push_back(make_mov_imm("r_pos", 0));              // PC=1: pos=0
    stmts.push_back(make_mov_imm("r_len", 8));              // PC=2: len=8
    stmts.push_back(make_bfe_u32("r2", "r1", "r_pos", "r_len")); // PC=3
    stmts.push_back(make_ret());                            // PC=4

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    int ret_pc = -1;
    for (int step = 0; step < 16; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 4) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 4);

    // Extract low 8 bits of each lane_id: 0..31 stays unchanged for u32.
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = get_reg_u32(w, "r2", lane);
        uint32_t expected = static_cast<uint32_t>(lane) & 0xFFu;
        INFO("lane " << lane << " r2 = 0x" << std::hex << v);
        CHECK(v == expected);
    }
}

TEST_CASE("integration_ptx_bitwise_bfe_upper_byte",
          "[integration][ptx][bitwise][bfe][regression]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(4);
    stmts.push_back(make_mov_imm("r_pos", 24));             // PC=0: pos=24
    stmts.push_back(make_mov_imm("r_len", 8));              // PC=1: len=8
    stmts.push_back(make_bfe_u32("r2", "r1", "r_pos", "r_len")); // PC=2
    stmts.push_back(make_ret());                            // PC=3

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    // Pre-load r1 with a value where the upper byte is distinguishable
    // from the lower byte, so we can assert pos actually shifts.
    set_reg_per_lane_u32(
        w, "r1", [](int lane) { return static_cast<uint32_t>(0xAB000000u | lane); });

    int ret_pc = -1;
    for (int step = 0; step < 16; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 3) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 3);

    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = get_reg_u32(w, "r2", lane);
        uint32_t expected = 0xABu; // only the upper byte, masked to 8 bits
        INFO("lane " << lane << " r2 = 0x" << std::hex << v);
        CHECK(v == expected);
    }
}

TEST_CASE("integration_ptx_bitwise_bfe_len_zero",
          "[integration][ptx][bitwise][bfe][regression]") {
    // Edge case: len=0 must produce result=0 (per PTX ISA spec).
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(4);
    stmts.push_back(make_mov_imm("r_pos", 0));              // PC=0
    stmts.push_back(make_mov_imm("r_len", 0));              // PC=1: len=0
    stmts.push_back(make_bfe_u32("r2", "r1", "r_pos", "r_len")); // PC=2
    stmts.push_back(make_ret());                            // PC=3

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(
        w, "r1", [](int) { return 0xDEADBEEFu; });

    int ret_pc = -1;
    for (int step = 0; step < 16; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 3) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 3);

    for (int lane = 0; lane < 32; ++lane) {
        CHECK(get_reg_u32(w, "r2", lane) == 0u);
    }
}

TEST_CASE("integration_ptx_bitwise_bfe_pos_exceeds_width",
          "[integration][ptx][bitwise][bfe][regression]") {
    // Edge case: pos >= width must produce result=0 (per PTX ISA spec).
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(4);
    stmts.push_back(make_mov_imm("r_pos", 32));             // PC=0: pos=32 (>= 32-bit width)
    stmts.push_back(make_mov_imm("r_len", 8));              // PC=1
    stmts.push_back(make_bfe_u32("r2", "r1", "r_pos", "r_len")); // PC=2
    stmts.push_back(make_ret());                            // PC=3

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(
        w, "r1", [](int) { return 0xFFFFFFFFu; });

    int ret_pc = -1;
    for (int step = 0; step < 16; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 3) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 3);

    for (int lane = 0; lane < 32; ++lane) {
        CHECK(get_reg_u32(w, "r2", lane) == 0u);
    }
}
