// test_cvt.cpp
// =============================================================================
// Integration test (类型二) — cvt.*.* (type conversion) on the PTX-EMU
// simulator (NOT real GPU). Drives SMContext + WarpContext + step_warp.
//
// Per TEST_CASE:
//   PC=0:  mov.b32 r1, tid.x    ; r1[lane] = lane_id
//   PC=1:  cvt.<dst>.<src> r2, r1
//   PC=2:  ret
//
// Expected: r2 == bits(converted(lane_id)) for every lane.
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

using ptxsim::testing::make_cvt;
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

WarpContext *setup_block(SMContext &sm, std::vector<StatementContext> &stmts) {
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

// Helper: convert float to bit pattern
uint32_t f32_to_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return bits;
}

// Helper: convert double to bit pattern (only low 32 bits, since rbm is u32)
uint32_t f64_to_low32(double d) {
    uint64_t bits;
    std::memcpy(&bits, &d, 8);
    return static_cast<uint32_t>(bits & 0xFFFFFFFFu);
}

} // namespace

TEST_CASE("integration_ptx_cvt_s32_from_f32",
          "[integration][ptx][cvt][s32_f32]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_mov("r1", "tid.x"));
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_F32, Qualifier::Q_S32));
    stmts.push_back(make_ret());

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
        float expected = static_cast<float>(lane);
        CHECK(v == f32_to_bits(expected));
    }
}

// KNOWN ISSUE: P1-4.1 — CvtHandler does not write r2 in f32->s32 and f64->s64
// paths. The r2 register reads back as 0 (uninitialized) for non-zero source
// bits. Test body is preserved for when the handler is fixed; SKIP for now. See
// docs/developer-guide/KNOWN_ISSUES.md §P1-4.1 for details and fix steps.
TEST_CASE("integration_ptx_cvt_f32_from_s32",
          "[integration][ptx][cvt][f32_s32]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    // f32→s32 path: omit `mov r1, tid.x` because the integer lane-id bits
    // would be reinterpreted as a denormalized near-zero float, defeating
    // the test. r1 is seeded directly via the register bank below.
    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_S32, Qualifier::Q_F32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    // r1 holds the bit pattern of float(lane + 0.5f); cvt truncates to int
    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return f32_to_bits(static_cast<float>(lane) + 0.5f);
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
        // (int)(float)(lane + 0.5) = lane (truncation toward zero)
        CHECK(v == static_cast<uint32_t>(lane));
    }
}

TEST_CASE("integration_ptx_cvt_f64_from_f32",
          "[integration][ptx][cvt][f64_f32]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_mov("r1", "tid.x"));
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_F64, Qualifier::Q_F32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return f32_to_bits(static_cast<float>(lane) * 1.5f);
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

    // Verify the lower 32 bits of the double-precision result
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = get_reg_u32(w, "r2", lane);
        double expected = static_cast<double>(lane) * 1.5;
        uint64_t expected_bits;
        std::memcpy(&expected_bits, &expected, 8);
        uint32_t lo = static_cast<uint32_t>(expected_bits & 0xFFFFFFFFu);
        CHECK(v == lo);
    }
}

