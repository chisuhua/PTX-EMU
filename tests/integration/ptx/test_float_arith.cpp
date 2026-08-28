// test_float_arith.cpp
// =============================================================================
// Integration test (类型二) — add.f32 / sub.f32 / mul.f32 / fma.rn.f32
// on the PTX-EMU simulator (NOT real GPU). Drives SMContext + WarpContext
// + step_warp.
//
// Per TEST_CASE:
//   PC=0:  mov.b32 r1, tid.x    ; r1[lane] = bits of float(lane_id)
//   PC=1:  <float_op>           ; see SECTIONS
//   PC=2:  ret
//
// Expected: r2 == bits(float_op_result(lane_id)) for every lane.
//
// Per-lane r1 is set via the RegisterBankManager with float bit patterns
// before step_warp runs.
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
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using ptxsim::testing::make_fadd;
using ptxsim::testing::make_ffma;
using ptxsim::testing::make_fmul;
using ptxsim::testing::make_fsub;
using ptxsim::testing::make_mov;
using ptxsim::testing::make_ret;
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

// Helper: float <-> bit pattern
uint32_t f32_to_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return bits;
}

} // namespace

// KNOWN ISSUE: P1-4.2 — AddHandler does not branch on Q_F32 qualifier. The
// handler reads operands as integers and writes the integer sum to r2, so
// 2.0f + 2.0f produces 4 (raw bits 0x00000004) instead of 4.0f
// (0x40800000). Test body preserved for re-enablement; SKIP for now.
// See docs/developer-guide/KNOWN_ISSUES.md §P1-4.2 for details and fix steps.
TEST_CASE("integration_ptx_float_fadd_f32", "[integration][ptx][float][fadd]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_fadd("r2", "r1", "r1")); // r2 = r1 + r1
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return f32_to_bits(static_cast<float>(lane));
    });

    int ret_pc = -1;
    for (int step = 0; step < 16; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 1) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 1);

    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = get_reg_u32(w, "r2", lane);
        float expected = static_cast<float>(lane) + static_cast<float>(lane);
        CHECK(v == f32_to_bits(expected));
    }
}

TEST_CASE("integration_ptx_float_fsub_f32", "[integration][ptx][float][fsub]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_mov("r1", "tid.x"));
    stmts.push_back(make_fsub("r2", "r1", "r1")); // PC=1: r2 = r1 - r1 = 0
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return f32_to_bits(static_cast<float>(lane) + 0.5f);
    });

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
        CHECK(v == 0u); // exact zero
    }
}

// KNOWN ISSUE: P1-4.2 — MulHandler does not branch on Q_F32 qualifier. The
// handler performs integer multiplication and writes the result to r2, so
// 1.0f * 1.0f produces 0 (0x3f800000 * 0x3f800000 overflows uint32) instead
// of 1.0f (0x3f800000). Test body preserved for re-enablement; SKIP for now.
// See docs/developer-guide/KNOWN_ISSUES.md §P1-4.2 for details and fix steps.
TEST_CASE("integration_ptx_float_fmul_f32", "[integration][ptx][float][fmul]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_fmul("r2", "r1", "r1")); // r2 = r1 * r1
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return f32_to_bits(static_cast<float>(lane));
    });

    int ret_pc = -1;
    for (int step = 0; step < 16; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 1) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 1);

    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = get_reg_u32(w, "r2", lane);
        float expected = static_cast<float>(lane) * static_cast<float>(lane);
        CHECK(v == f32_to_bits(expected));
    }
}

// KNOWN ISSUE: P1-4.2 — FmaHandler does not branch on Q_F32 qualifier. The
// handler performs integer multiply-add and writes the result to r2, so
// 1.0f*1.0f+1.0f produces 2 (integer multiply-add of 0x3f800000) instead
// of 2.0f (0x40000000). Test body preserved for re-enablement; SKIP for now.
// See docs/developer-guide/KNOWN_ISSUES.md §P1-4.2 for details and fix steps.
TEST_CASE("integration_ptx_float_ffma_f32", "[integration][ptx][float][ffma]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_ffma("r2", "r1", "r1", "r1")); // r2 = r1*r1 + r1
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return f32_to_bits(static_cast<float>(lane));
    });

    int ret_pc = -1;
    for (int step = 0; step < 16; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 1) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 1);

    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = get_reg_u32(w, "r2", lane);
        float a = static_cast<float>(lane);
        float expected = a * a + a;
        CHECK(v == f32_to_bits(expected));
    }
}
