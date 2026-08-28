// test_cvt_float_to_float.cpp
// =============================================================================
// Integration test (类型二) — float→float CVT (6 TEST_CASEs)
//
// Covers f16/f32/f64 mutual conversions
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

TEST_CASE("integration_ptx_cvt_f32_to_f32_identity",
          "[integration][ptx][cvt][f2f]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_F32, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return f32_bits(static_cast<float>(lane) * 0.5f);
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) ==
                f32_bits(static_cast<float>(lane) * 0.5f));
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_f32_narrowing_self",
          "[integration][ptx][cvt][f2f]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_F32, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return f32_bits(static_cast<float>(lane) * 1.5f);
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        float expected = static_cast<float>(lane) * 1.5f;
        REQUIRE(get_reg_u32(w, "r2", lane) == f32_bits(expected));
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_f64_widening",
          "[integration][ptx][cvt][f2f]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_F64, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int lane) { return f32_bits(1.0f); });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        // f32(1.0f) = 0x3F800000, widened f64 = 0x3FF0000000000000
        // Low 32 bits = 0x00000000 (zero because mantissa is 0)
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}

TEST_CASE("integration_ptx_cvt_f64_to_f64_identity",
          "[integration][ptx][cvt][f2f]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_F64, ptxemu::ir::Qualifier::Q_F64));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int lane) {
        double d = 2.718281828;
        uint64_t bits;
        std::memcpy(&bits, &d, 8);
        return static_cast<uint32_t>(bits & 0xFFFFFFFFu);
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        double expected = 2.718281828;
        uint64_t exp_bits;
        std::memcpy(&exp_bits, &expected, 8);
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(exp_bits));
    }
}

TEST_CASE("integration_ptx_cvt_f16_to_f32_via_half_utils",
          "[integration][ptx][cvt][f2f][half]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_F32, ptxemu::ir::Qualifier::Q_F16));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return static_cast<uint32_t>(0x3C00); // 1.0 in half
    });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == f32_bits(1.0f));
    }
}

TEST_CASE("integration_ptx_cvt_f32_to_f16_via_half_utils",
          "[integration][ptx][cvt][f2f][half]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_cvt("r2", "r1", ptxemu::ir::Qualifier::Q_F16, ptxemu::ir::Qualifier::Q_F32));
    stmts.push_back(make_ret());
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    set_reg_per_lane_u32(w, "r1", [](int lane) { return f32_bits(2.0f); });
    for (int s = 0; s < 16; ++s)
        if (step_warp(w, stmts) == 1)
            break;
    for (int lane = 0; lane < 32; ++lane) {
        // 2.0 in half = 0x4000
        uint32_t v = get_reg_u32(w, "r2", lane);
        REQUIRE((v & 0xFFFF) == 0x4000);
    }
}
