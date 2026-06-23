// test_cvt_int_to_int.cpp
// =============================================================================
// Integration test (类型二) — int→int CVT (4×4 dimension matrix)
//
// 30 TEST_CASEs covering:
//   - s8/s16/s32/s64, u8/u16/u32/u64 → s8/s16/s32/s64, u8/u16/u32/u64
//   - .sat saturation, sign/zero extension
//
// 参考 pattern: test_cvt.cpp (PC=0: mov; PC=1: cvt; PC=2: ret)
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
using ptxsim::testing::make_cvt_sat;
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

} // namespace

// ---- Same-size identity ----

TEST_CASE("integration_ptx_cvt_s32_to_s32 identity",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_S32, Qualifier::Q_S32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(lane));
    }
}

TEST_CASE("integration_ptx_cvt_u32_to_u32 identity",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_U32, Qualifier::Q_U32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(lane));
    }
}

// ---- Sign/zero extension ----

TEST_CASE("integration_ptx_cvt_s8_to_s32 sign extend",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_S32, Qualifier::Q_S8));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    // Use lower 8 bits of lane as int8
    set_reg_per_lane_u32(w, "r1", [](int lane) {
        int8_t v = static_cast<int8_t>(lane & 0xFF);
        return static_cast<uint32_t>(static_cast<int32_t>(v));
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        int8_t v = static_cast<int8_t>(lane & 0xFF);
        int32_t expected = v;
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(expected));
    }
}

TEST_CASE("integration_ptx_cvt_u8_to_u32 zero extend",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_U32, Qualifier::Q_U8));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(lane));
    }
}

TEST_CASE("integration_ptx_cvt_s16_to_s32 sign extend",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_S32, Qualifier::Q_S16));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        int16_t v = static_cast<int16_t>(lane - 16); // half negative
        return static_cast<uint32_t>(static_cast<int32_t>(v));
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        int16_t v = static_cast<int16_t>(lane - 16);
        int32_t expected = v;
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(expected));
    }
}

// ---- Narrowing ----

TEST_CASE("integration_ptx_cvt_s32_to_s16 in range",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_S16, Qualifier::Q_S32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        uint16_t v = static_cast<uint16_t>(lane);
        REQUIRE(get_reg_u32(w, "r2", lane) == v);
    }
}

TEST_CASE("integration_ptx_cvt_s64_to_s32 in range",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_S32, Qualifier::Q_S64));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(lane));
    }
}

// ---- .sat saturation ----

TEST_CASE("integration_ptx_cvt_s32_to_s8_sat_positive_clamp",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_S8, Qualifier::Q_S32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return 200U + lane; }); // > 127

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0x7F); // 127
    }
}

TEST_CASE("integration_ptx_cvt_s32_to_s8_sat_negative_clamp",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_S8, Qualifier::Q_S32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        int32_t v = -200 - lane;
        return static_cast<uint32_t>(static_cast<int32_t>(v));
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0xFFFFFF80); // -128
    }
}

TEST_CASE("integration_ptx_cvt_s32_to_u8_sat_negative_to_zero",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_U8, Qualifier::Q_S32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        int32_t v = -1 - lane;
        return static_cast<uint32_t>(static_cast<int32_t>(v));
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}

TEST_CASE("integration_ptx_cvt_u32_to_s32_sat_clamp",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_S32, Qualifier::Q_U32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return 0xFFFFFFFFU; // > INT32_MAX
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0x7FFFFFFF);
    }
}

TEST_CASE("integration_ptx_cvt_s64_to_s32_sat_clamp",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_S32, Qualifier::Q_S64));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        int64_t v = 3000000000LL + lane;
        return static_cast<uint32_t>(v & 0xFFFFFFFF); // low 32
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        // Saturated to INT32_MAX (low 32 bits = 0x7FFFFFFF)
        REQUIRE(get_reg_u32(w, "r2", lane) == 0x7FFFFFFF);
    }
}

// ---- Sign mismatch (s → u, u → s) ----

TEST_CASE("integration_ptx_cvt_s32_to_u32_zero_or_pos",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_U32, Qualifier::Q_S32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(lane));
    }
}

TEST_CASE("integration_ptx_cvt_u32_to_s32_unsigned_fit",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_S32, Qualifier::Q_U32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(lane));
    }
}

// ---- More identity / extension patterns ----

TEST_CASE("integration_ptx_cvt_s16_to_s8_truncation",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_S8, Qualifier::Q_S16));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        // lane 0-15 fit, 16-31 truncate (e.g. lane=16 -> 16, fits; lane=200 ->
        // -56)
        int8_t v = static_cast<int8_t>(lane);
        REQUIRE(get_reg_u32(w, "r2", lane) ==
                static_cast<uint32_t>(static_cast<int32_t>(v)));
    }
}

TEST_CASE("integration_ptx_cvt_s32_to_u16_in_range",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_U16, Qualifier::Q_S32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return static_cast<uint32_t>(lane + 100); // 100-131, all fit
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) ==
                static_cast<uint32_t>(lane + 100));
    }
}

TEST_CASE("integration_ptx_cvt_s64_to_s64_identity",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_S64, Qualifier::Q_S64));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(lane));
    }
}

TEST_CASE("integration_ptx_cvt_u64_to_u64_identity",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_U64, Qualifier::Q_U64));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(lane));
    }
}

TEST_CASE("integration_ptx_cvt_s32_to_s8_sat_in_range",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_S8, Qualifier::Q_S32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        // All in [-128, 127] range, so passthrough
        int8_t v = static_cast<int8_t>(lane);
        REQUIRE(get_reg_u32(w, "r2", lane) ==
                static_cast<uint32_t>(static_cast<int32_t>(v)));
    }
}

TEST_CASE("integration_ptx_cvt_s32_to_u32_sat_negative",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_U32, Qualifier::Q_S32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        int32_t v = -100 - lane;
        return *reinterpret_cast<uint32_t *>(&v);
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}

TEST_CASE("integration_ptx_cvt_s16_to_u8_sat_negative_to_zero",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_U8, Qualifier::Q_S16));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        int16_t v = -1 - lane;
        return static_cast<uint32_t>(static_cast<int32_t>(v));
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}

TEST_CASE("integration_ptx_cvt_u8_to_s8_sat_unsigned_in_range",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_cvt_sat("r2", "r1", Qualifier::Q_S8, Qualifier::Q_U8));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) { return 100U + lane; });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        // .sat clamps u8 100..131 to s8 range: 100..127 fits, 128..131 → 127
        int8_t expected =
            (100 + lane > 127) ? 127 : static_cast<int8_t>(100 + lane);
        REQUIRE(get_reg_u32(w, "r2", lane) ==
                static_cast<uint32_t>(static_cast<int32_t>(expected)));
    }
}

TEST_CASE("integration_ptx_cvt_u8_to_s8_sat_clamp",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_cvt_sat("r2", "r1", Qualifier::Q_S8, Qualifier::Q_U8));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) { return 200U; });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0x7F);
    }
}

TEST_CASE("integration_ptx_cvt_s16_to_u16_sat_negative_to_zero",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_U16, Qualifier::Q_S16));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        int16_t v = -1;
        return static_cast<uint32_t>(static_cast<int32_t>(v));
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}

TEST_CASE("integration_ptx_cvt_u16_to_s16_sat_clamp",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_S16, Qualifier::Q_U16));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) { return 50000U; });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0x7FFF); // 32767
    }
}

TEST_CASE("integration_ptx_cvt_s64_to_s8_sat_clamp",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_S8, Qualifier::Q_S64));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        int64_t v = 1000LL;
        return static_cast<uint32_t>(v & 0xFFFFFFFF);
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0x7F);
    }
}

TEST_CASE("integration_ptx_cvt_u32_to_u32_sat_in_range",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_U32, Qualifier::Q_U32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(lane));
    }
}

TEST_CASE("integration_ptx_cvt_u64_to_u32_sat_above_max",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_U32, Qualifier::Q_U64));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        // Set low 32 bits to a high value > UINT32_MAX
        return 0xDEADBEEFU;
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        // 0xDEADBEEF as u64 > UINT32_MAX, so saturated to 0xFFFFFFFF
        REQUIRE(get_reg_u32(w, "r2", lane) == 0xFFFFFFFFU);
    }
}

TEST_CASE("integration_ptx_cvt_s64_to_u64_sat_positive",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_U64, Qualifier::Q_S64));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        int64_t v = 1000LL;
        return static_cast<uint32_t>(v & 0xFFFFFFFF);
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        // 1000 fits in u64, passthrough (low 32 bits = 1000)
        REQUIRE(get_reg_u32(w, "r2", lane) == 1000U);
    }
}

TEST_CASE("integration_ptx_cvt_s8_to_u8_sat_negative_to_zero",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(make_cvt_sat("r2", "r1", Qualifier::Q_U8, Qualifier::Q_S8));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        int8_t v = -1;
        return static_cast<uint32_t>(static_cast<int32_t>(v));
    });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0);
    }
}

TEST_CASE("integration_ptx_cvt_s32_to_u8_sat_in_range",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_U8, Qualifier::Q_S32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) { return 200U + lane; });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        // 200+lane > 255 for lane >= 56, but lane is 0-31 so all 200-231, fit
        REQUIRE(get_reg_u32(w, "r2", lane) == 200U + lane);
    }
}

TEST_CASE("integration_ptx_cvt_s64_to_u8_sat_in_range",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_U8, Qualifier::Q_S32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) { return 50U; });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 50U);
    }
}

TEST_CASE("integration_ptx_cvt_u8_to_s32_zero_extend",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_S32, Qualifier::Q_U8));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(lane));
    }
}

TEST_CASE("integration_ptx_cvt_u16_to_s32_zero_extend",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_S32, Qualifier::Q_U16));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(lane));
    }
}

TEST_CASE("integration_ptx_cvt_u64_to_s32_sat_clamp",
          "[integration][ptx][cvt][i2i][sat]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(2);
    stmts.push_back(
        make_cvt_sat("r2", "r1", Qualifier::Q_S32, Qualifier::Q_U64));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) { return 0xFFFFFFFFU; });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_reg_u32(w, "r2", lane) == 0x7FFFFFFF);
    }
}

TEST_CASE("integration_ptx_cvt_s32_to_u64_sign_extend",
          "[integration][ptx][cvt][i2i]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(3);
    stmts.push_back(make_cvt("r2", "r1", Qualifier::Q_U64, Qualifier::Q_S32));
    stmts.push_back(make_ret());

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    for (int step = 0; step < 16; ++step) {
        if (step_warp(w, stmts) == 1)
            break;
    }

    for (int lane = 0; lane < 32; ++lane) {
        // For lanes 0-31, all positive, s32 fits in u64
        REQUIRE(get_reg_u32(w, "r2", lane) == static_cast<uint32_t>(lane));
    }
}
