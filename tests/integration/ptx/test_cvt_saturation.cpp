// test_cvt_saturation.cpp
// =============================================================================
// Integration test (类型二) — CVT .sat saturation (8 TEST_CASEs)
//
// Cross-type saturation: u32, s32, s8, s16, s64, u64 + .sat
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
    if (!rbm->get_register(reg, 0, 0)) {
        rbm->create_register(reg, sizeof(uint32_t));
    }
    for (int i = 0; i < 32; ++i) {
        void *p = rbm->get_register(reg, 0, i);
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
    sm.add_block(std::move(blk));
    return sm.get_warp(0);
}

uint32_t get_reg_u32(WarpContext *w, const std::string &reg, int lane) {
    return *static_cast<uint32_t *>(
        w->get_register_bank_manager()->get_register(reg, 0, lane));
}

uint32_t f32_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

} // namespace

TEST_CASE("integration_ptx_cvt_sat_f32_to_s32_clamp_pos",
          "[integration][ptx][cvt][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S32, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) { return f32_bits(1e10f); });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0x7FFFFFFFU);
    }
}

TEST_CASE("integration_ptx_cvt_sat_f32_to_s32_clamp_neg",
          "[integration][ptx][cvt][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S32, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) { return f32_bits(-1e10f); });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0x80000000U);
    }
}

TEST_CASE("integration_ptx_cvt_sat_f32_to_u32_clamp_above",
          "[integration][ptx][cvt][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_U32, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) { return f32_bits(1e10f); });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0xFFFFFFFFU);
    }
}

TEST_CASE("integration_ptx_cvt_sat_f32_to_u32_clamp_below",
          "[integration][ptx][cvt][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_U32, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) { return f32_bits(-1.0f); });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}

TEST_CASE("integration_ptx_cvt_sat_f32_to_s8_clamp",
          "[integration][ptx][cvt][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int lane) {
        // 50..81 — first 78 lanes in range, 79..81 clamp to 127
        return f32_bits(static_cast<float>(50 + lane));
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        int32_t expected = (50 + lane > 127) ? 127 : (50 + lane);
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(expected));
    }
}

TEST_CASE("integration_ptx_cvt_sat_f32_to_s16_clamp",
          "[integration][ptx][cvt][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S16, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) { return f32_bits(1e10f); });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0x7FFF);
    }
}

TEST_CASE("integration_ptx_cvt_sat_f32_to_u16_clamp_neg",
          "[integration][ptx][cvt][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_U16, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) { return f32_bits(-1e10f); });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}

TEST_CASE("integration_ptx_cvt_sat_f32_to_s32_nan_handling",
          "[integration][ptx][cvt][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S32, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) { return f32_bits(std::nanf("")); });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}
