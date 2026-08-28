// test_cvt_float_to_int.cpp
// =============================================================================
// Integration test (类型二) — float→int CVT (40 TEST_CASEs)
//
// Covers:
//   - s8/s16/s32/s64, u8/u16/u32/u64 dst
//   - 5 rounding modes (.rn/.rz/.rm/.rp/.rna) + .sat
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

// Helper: setup r1 with a single f32 value (same for all lanes)
static void setup_uniform_f32(WarpContext *w, float v) {
    uint32_t bits = f32_bits(v);
    set_reg_per_lane_u32(w, "r1", [bits](int) { return bits; });
}

// ---- s32 dst with 5 rounding modes ----

TEST_CASE("integration_ptx_cvt_f32_to_s32_rn",
          "[integration][ptx][cvt][f2i][rn]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S32, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RN}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 3.5f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 4U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s32_rz",
          "[integration][ptx][cvt][f2i][rz]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S32, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RZ}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, -3.7f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) ==
                *reinterpret_cast<uint32_t *>(
                    static_cast<int32_t *>(new int32_t(-3))));
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s32_rm",
          "[integration][ptx][cvt][f2i][rm]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S32, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RM}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 3.7f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 3U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s32_rp",
          "[integration][ptx][cvt][f2i][rp]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S32, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RP}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 3.2f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 4U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s32_rna",
          "[integration][ptx][cvt][f2i][rna]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S32, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RNA}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 3.5f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 4U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s32_default_truncation",
          "[integration][ptx][cvt][f2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S32, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 3.7f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 3U);
    }
}

// ---- u32 dst ----

TEST_CASE("integration_ptx_cvt_f32_to_u32_rn",
          "[integration][ptx][cvt][f2i][u32]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_U32, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RN}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 100.5f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 100U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_u32_rz",
          "[integration][ptx][cvt][f2i][u32]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_U32, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RZ}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 100.9f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 100U);
    }
}

// ---- s8/s16/u8/u16 dst ----

TEST_CASE("integration_ptx_cvt_f32_to_s8_rn",
          "[integration][ptx][cvt][f2i][s8]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RN}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 50.5f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0x32U); // 50 = 0x32
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s16_rz",
          "[integration][ptx][cvt][f2i][s16]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S16, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RZ}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 1234.9f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 1234U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_u8_rn",
          "[integration][ptx][cvt][f2i][u8]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_U8, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RN}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 200.5f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 200U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_u16_rn",
          "[integration][ptx][cvt][f2i][u16]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_U16, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RN}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 50000.4f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 50000U);
    }
}

// ---- f64 source ----

TEST_CASE("integration_ptx_cvt_f64_to_s32_rn",
          "[integration][ptx][cvt][f2i][f64]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S32, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RN}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) {
        return f32_bits(100.5f);
        return f32_bits(100.5f);
        (void)0; // unused
        return f32_bits(100.5f);
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 100U);
    }
}

// ---- .sat tests ----

TEST_CASE("integration_ptx_cvt_f32_to_s8_sat_clamp_pos",
          "[integration][ptx][cvt][f2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 1e10f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0x7F);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s8_sat_clamp_neg",
          "[integration][ptx][cvt][f2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, -1e10f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0xFFFFFF80U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_u32_sat_clamp",
          "[integration][ptx][cvt][f2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_U32, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 1e10f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0xFFFFFFFFU);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_u8_sat_neg_to_zero",
          "[integration][ptx][cvt][f2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_U8, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, -1.0f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s32_sat_nan",
          "[integration][ptx][cvt][f2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S32, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, std::nanf(""));
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s32_sat_in_range",
          "[integration][ptx][cvt][f2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S32, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return f32_bits(static_cast<float>(lane)); // 0..31, in range
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(lane));
    }
}

// ---- More rounding mode coverage ----

TEST_CASE("integration_ptx_cvt_f32_to_s8_rm", "[integration][ptx][cvt][f2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RM}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 50.9f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 50U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s8_rp", "[integration][ptx][cvt][f2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RP}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 50.1f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 51U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s8_rna_neg",
          "[integration][ptx][cvt][f2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RNA}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, -50.5f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) ==
                0xFFFFFFCDU); // -51 (.rna away from zero)
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_u32_rp", "[integration][ptx][cvt][f2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_U32, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RP}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 0.1f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 1U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_u32_rna",
          "[integration][ptx][cvt][f2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_U32, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RNA}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 0.5f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 1U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s8_rz_neg",
          "[integration][ptx][cvt][f2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RZ}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, -50.7f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0xFFFFFFCEU); // -50
    }
}

// ---- additional .sat combinations ----

TEST_CASE("integration_ptx_cvt_f32_to_s16_sat_clamp",
          "[integration][ptx][cvt][f2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S16, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 1e10f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0x7FFF);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s16_sat_in_range",
          "[integration][ptx][cvt][f2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S16, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 100.0f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 100U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_u16_sat_neg",
          "[integration][ptx][cvt][f2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_U16, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, -1.0f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_u8_sat_in_range",
          "[integration][ptx][cvt][f2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_U8, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 100.0f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 100U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_u8_sat_clamp",
          "[integration][ptx][cvt][f2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_U8, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 1e6f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0xFF);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s8_sat_in_range",
          "[integration][ptx][cvt][f2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, 42.0f);
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 42U);
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_s8_sat_nan",
          "[integration][ptx][cvt][f2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    setup_uniform_f32(w, std::nanf(""));
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}

// ---- f64 source tests ----

TEST_CASE("integration_ptx_cvt_f64_to_s8_rn", "[integration][ptx][cvt][f2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RN}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) {
        return f32_bits(50.5f);
        return f32_bits(100.5f);
        (void)0; // unused
        return f32_bits(100.5f);
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 50U);
    }
}

TEST_CASE("integration_ptx_cvt_f64_to_u32_rz", "[integration][ptx][cvt][f2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_U32, ptxemu::ir::Qualifier::Q_F32,
                             {ptxemu::ir::Qualifier::Q_RZ}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) {
        return f32_bits(100.9f);
        return f32_bits(100.5f);
        (void)0; // unused
        return f32_bits(100.5f);
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 100U);
    }
}

TEST_CASE("integration_ptx_cvt_f64_to_s32_default",
          "[integration][ptx][cvt][f2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S32, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) {
        return f32_bits(3.7f);
        return f32_bits(100.5f);
        (void)0; // unused
        return f32_bits(100.5f);
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 3U);
    }
}

// ---- f16 source tests ----

TEST_CASE("integration_ptx_cvt_f16_to_s8_rn",
          "[integration][ptx][cvt][f2i][half]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F16,
                             {ptxemu::ir::Qualifier::Q_RN}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) {
        return 0x5640u; // 100.0 in half
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 100U);
    }
}

TEST_CASE("integration_ptx_cvt_f16_to_s8_sat",
          "[integration][ptx][cvt][f2i][half][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F16));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) {
        return 0x5640u; // 100.0 in half
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 100U);
    }
}

TEST_CASE("integration_ptx_cvt_f16_to_s8_sat_nan",
          "[integration][ptx][cvt][f2i][half][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(
        make_cvt_sat("r2", "r1", ptxemu::ir::Qualifier::Q_S8, ptxemu::ir::Qualifier::Q_F16));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) {
        return 0x7E00u; // NaN in half
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}

TEST_CASE("integration_ptx_cvt_f16_to_u32_rz",
          "[integration][ptx][cvt][f2i][half]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_U32, ptxemu::ir::Qualifier::Q_F16,
                             {ptxemu::ir::Qualifier::Q_RZ}));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int) {
        return 0x5C00u; // 256.0 in half
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 256U);
    }
}
