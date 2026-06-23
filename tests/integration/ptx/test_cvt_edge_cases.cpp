// test_cvt_edge_cases.cpp
// =============================================================================
// Integration test (类型二) — cvt.*.* 边缘场景 PTX 端到端验证
//
// 覆盖:
//   1. cvt.f32.f16 denormal round-trip: 验证 half denormal → f32 正确性
//   2. cvt.u32.f32.sat boundary equality: temp == 4294967295.0f 应饱和
//   3. cvt.u32.f32.sat above boundary: 远超 UINT32_MAX 的值应饱和到 0xFFFFFFFF
//
// 历史: T2-6 Step 1 (commit d3c77b5) 保留 2 个 pre-existing bug,
// 此次 commit 修复后用此测试做端到端验证。
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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using ptxsim::testing::make_cvt;
using ptxsim::testing::make_cvt_sat;
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

uint32_t f32_to_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return bits;
}

} // namespace

TEST_CASE("integration_ptx_cvt_f32_f16_denormal_smallest",
          "[integration][ptx][cvt][edge_case][denormal]") {
    // The smallest positive denormal half value (0x0001 = 2^-24) must
    // convert to a small finite float (2^-24 ≈ 5.96e-8), not a huge
    // value (the pre-fix bug returned ~5e30).
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_F32, Qualifier::Q_F16));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int /*lane*/) {
        // Low 16 bits hold the half value 0x0001; high 16 bits unused by
        // the cvt.f32.f16 path (it reads 2 bytes from src).
        return static_cast<uint32_t>(0x0001u);
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

    const uint32_t expected_bits = f32_to_bits(5.9604644775390625e-08f);
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = get_reg_u32(w, "r2", lane);
        CHECK(v == expected_bits);
    }
}

TEST_CASE("integration_ptx_cvt_u32_f32_rpi_boundary_equality",
          "[integration][ptx][cvt][edge_case][sat]") {
    // 4294967295.0f in float32 rounds up to 4294967296.0f, which is just
    // above UINT32_MAX. With cvt.u32.f32.rpi (round toward +inf), the
    // conversion path calls should_saturate_uint32(temp, 4294967295.0f)
    // to decide whether to clamp to UINT32_MAX. Pre-fix the helper used
    // strict `<` so the upper-bound check returned false at the
    // boundary and the value fell through to static_cast<uint32_t>(temp)
    // (undefined behavior, often 0 on x86). Post-fix the helper uses
    // `<=`, so the boundary value saturates correctly to 0xFFFFFFFF.
    //
    // We use .rpi here (not .sat) because the plain .sat path in
    // arithmetic_conversion.cpp has its own `temp > 4294967295.0f`
    // check, which has a separate (out-of-scope) precision bug —
    // see tasks.md "Out of Scope" + follow-up audit. The .rpi path
    // actually exercises the should_saturate_uint32 fix.
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_U32, Qualifier::Q_F32,
                             {Qualifier::Q_RPI}));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(
        w, "r1", [](int /*lane*/) { return f32_to_bits(4294967295.0f); });

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
        CHECK(v == 0xFFFFFFFFu);
    }
}

TEST_CASE("integration_ptx_cvt_u32_f32_sat_above_boundary_clamps",
          "[integration][ptx][cvt][edge_case][sat]") {
    // A value way above UINT32_MAX (1e10f) must clamp to 0xFFFFFFFF under
    // .sat, regardless of the should_saturate_uint32 helper. The .sat path
    // uses a direct `temp > 4294967295.0f` check, so this is a separate
    // (already-correct) code path. We keep the test to guard against
    // future regressions in the .sat branch.
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_U32, Qualifier::Q_F32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int /*lane*/) { return f32_to_bits(1e10f); });

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
        CHECK(v == 0xFFFFFFFFu);
    }
}
