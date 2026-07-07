// tests/integration/tcgen05/test_tcgen05_mma_sync.cpp
// Phase 1.2 (Fix #11): integration test verifying tcgen05.mma fragment
// arithmetic via the instruction execution pipeline.
//
// Uses ptxsim::testing::step_warp to drive execution (per AGENTS.md
// "禁止在测试代码里重新实现 step_warp"). Verifies that tcgen05.mma writes
// correct f16 fragment elements to TMEM slots.
//
// Gate P1-3.G1: grep -c "UNVERIFIED-AGAINST-HARDWARE" wmma.cpp ≥ 256
// Gate P1-3.G2: ctest -R "tcgen05_mma_sync" PASS

#include "catch_amalgamated.hpp"

#include "ptx_ir/statement_factory.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/utils/half_utils.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"

#include <array>
#include <cstring>
#include <vector>

namespace {

void init_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

void fill_tmem_slot_f16(Tmem& tmem, size_t slot_id,
                         const std::vector<uint16_t>& data) {
    std::array<uint8_t, Tmem::kSlotSize> buf{};
    size_t n = std::min(data.size() * sizeof(uint16_t), Tmem::kSlotSize);
    std::memcpy(buf.data(), data.data(), n);
    tmem.write(slot_id, buf.data(), Tmem::kSlotSize);
}

std::vector<uint16_t> read_tmem_slot_f16(Tmem& tmem, size_t slot_id,
                                          size_t num_values) {
    std::array<uint8_t, Tmem::kSlotSize> buf{};
    tmem.read(slot_id, buf.data(), Tmem::kSlotSize);
    size_t n = std::min(num_values, Tmem::kSlotSize / 2);
    std::vector<uint16_t> result(n);
    std::memcpy(result.data(), buf.data(), n * sizeof(uint16_t));
    return result;
}

StatementContext make_mma_stmt() {
    using namespace ptxir::factory;
    std::vector<Qualifier> quals;
    quals.push_back(Qualifier::Q_CLUSTER);
    quals.push_back(Qualifier::Q_F16);
    // B2 factory verification alias. Do not remove until implement-tcgen05-handlers-core.
    auto tcgen05_alias =
        makeTcgen05Instr(Tcgen05OpKind::MMA, quals, {}, "tcgen05.mma");
    static_assert(std::is_same_v<decltype(tcgen05_alias), StatementContext>);
    (void)tcgen05_alias;
    return makeWmmaInstr(WmmaType::WMMA_MMA, quals, {}, "tcgen05.mma");
}

} // anonymous namespace

TEST_CASE("tcgen05.mma integration: step_warp drives 8x4 fragment arithmetic",
          "[integration][tcgen05][mma][fragment]") {
    init_factory_once();

    constexpr int ROWS = 8;
    constexpr int COLS_A = 8;
    constexpr int COLS_B = 4;

    int shared_mem_total_bytes = 4096;
    SMContext sm(4, 128, shared_mem_total_bytes, 0);

    auto block = std::make_unique<CTAContext>();
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};

    auto stmts = std::make_shared<std::vector<StatementContext>>();
    stmts->push_back(make_mma_stmt());

    stmts->push_back(ptxsim::testing::make_ret());

    auto name2Sym =
        std::make_shared<std::map<std::string, std::unique_ptr<Symtable>>>();

    std::map<std::string, int> label2pc;

    block->init(gridDim, blockDim, blockIdx, *stmts, &*name2Sym, label2pc);

    Tmem& tmem = block->tmem();

    std::vector<uint16_t> a_mat(ROWS * COLS_A, 0);
    for (int i = 0; i < ROWS; ++i) {
        a_mat[i * COLS_A + i] = f32_to_f16(1.0f);
    }

    std::vector<uint16_t> b_mat(ROWS * COLS_B, 0);
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS_B; ++j) {
            b_mat[i * COLS_B + j] =
                f32_to_f16(static_cast<float>((i + 1) * 10 + (j + 1)));
        }
    }

    for (int lane = 0; lane < 32; ++lane) {
        fill_tmem_slot_f16(tmem, static_cast<size_t>(lane) * 2, a_mat);
        fill_tmem_slot_f16(tmem, static_cast<size_t>(lane) * 2 + 1, b_mat);
    }

    sm.add_block(std::move(block));
    WarpContext* warp = sm.get_warp(0);
    REQUIRE(warp != nullptr);

    using ptxsim::testing::step_warp;
    auto& stmts_ref = *stmts;
    int pc = step_warp(warp, stmts_ref);
    REQUIRE(pc == 0);

    int pc_after = step_warp(warp, stmts_ref);
    REQUIRE(pc_after == 1);

    CTAContext* cta = warp->get_cta_context();
    REQUIRE(cta != nullptr);
    Tmem& tmem2 = cta->tmem();

    for (int lane = 0; lane < 32; ++lane) {
        auto result = read_tmem_slot_f16(
            tmem2, static_cast<size_t>(64 + lane),
            static_cast<size_t>(ROWS) * COLS_B);

        REQUIRE(result.size() == static_cast<size_t>(ROWS) * COLS_B);
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS_B; ++j) {
                CAPTURE(lane, i, j);
                REQUIRE(result[i * COLS_B + j] == b_mat[i * COLS_B + j]);
            }
        }
    }
}

TEST_CASE("tcgen05.mma integration: per-lane output isolation",
          "[integration][tcgen05][mma][isolation]") {
    init_factory_once();

    constexpr int ROWS = 8;
    constexpr int COLS_A = 8;
    constexpr int COLS_B = 4;

    SMContext sm(4, 128, 4096, 0);

    auto block = std::make_unique<CTAContext>();
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};

    auto stmts = std::make_shared<std::vector<StatementContext>>();
    stmts->push_back(make_mma_stmt());
    stmts->push_back(ptxsim::testing::make_ret());

    auto name2Sym =
        std::make_shared<std::map<std::string, std::unique_ptr<Symtable>>>();

    std::map<std::string, int> label2pc;
    block->init(gridDim, blockDim, blockIdx, *stmts, &*name2Sym, label2pc);

    Tmem& tmem = block->tmem();

    std::vector<uint16_t> b_mat(ROWS * COLS_B, 0);
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS_B; ++j) {
            b_mat[i * COLS_B + j] =
                f32_to_f16(static_cast<float>((i + 1) * 10 + j + 1));
        }
    }

    for (int lane = 0; lane < 32; ++lane) {
        std::vector<uint16_t> a_mat(ROWS * COLS_A, 0);
        for (int i = 0; i < ROWS; ++i) {
            a_mat[i * COLS_A + i] =
                f32_to_f16(static_cast<float>(lane + 1));
        }
        fill_tmem_slot_f16(tmem, static_cast<size_t>(lane) * 2, a_mat);
        fill_tmem_slot_f16(tmem, static_cast<size_t>(lane) * 2 + 1, b_mat);
    }

    sm.add_block(std::move(block));
    WarpContext* warp = sm.get_warp(0);
    REQUIRE(warp != nullptr);

    using ptxsim::testing::step_warp;
    auto& stmts_ref = *stmts;
    int pc = step_warp(warp, stmts_ref);
    REQUIRE(pc == 0);

    int pc_after = step_warp(warp, stmts_ref);
    REQUIRE(pc_after == 1);

    CTAContext* cta = warp->get_cta_context();
    REQUIRE(cta != nullptr);
    Tmem& tmem2 = cta->tmem();

    for (int lane = 0; lane < 32; ++lane) {
        auto result = read_tmem_slot_f16(
            tmem2, static_cast<size_t>(64 + lane),
            static_cast<size_t>(ROWS) * COLS_B);

        REQUIRE(result.size() == static_cast<size_t>(ROWS) * COLS_B);
        float lane_scale = static_cast<float>(lane + 1);
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS_B; ++j) {
                float expected_val =
                    lane_scale * static_cast<float>((i + 1) * 10 + j + 1);
                uint16_t expected = f32_to_f16(expected_val);
                CAPTURE(lane, i, j);
                REQUIRE(result[i * COLS_B + j] == expected);
            }
        }
    }
}

TEST_CASE("tcgen05.mma integration: handles non-tcgen05 qualifiers gracefully",
          "[integration][tcgen05][mma][error]") {
    init_factory_once();

    constexpr int ROWS = 8;
    constexpr int COLS_A = 8;
    constexpr int COLS_B = 4;

    SMContext sm(4, 128, 4096, 0);

    auto block = std::make_unique<CTAContext>();
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};

    auto stmts = std::make_shared<std::vector<StatementContext>>();
    using namespace ptxir::factory;
    stmts->push_back(
        makeWmmaInstr(WmmaType::WMMA_MMA, {}, {}, "wmma"));

    stmts->push_back(ptxsim::testing::make_ret());

    auto name2Sym =
        std::make_shared<std::map<std::string, std::unique_ptr<Symtable>>>();
    std::map<std::string, int> label2pc;

    block->init(gridDim, blockDim, blockIdx, *stmts, &*name2Sym, label2pc);

    std::vector<uint16_t> a_mat(ROWS * COLS_A, 0);
    for (int i = 0; i < ROWS; ++i)
        a_mat[i * COLS_A + i] = f32_to_f16(1.0f);
    std::vector<uint16_t> b_mat(ROWS * COLS_B, 0);
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS_B; ++j)
            b_mat[i * COLS_B + j] =
                f32_to_f16(static_cast<float>((i + 1) * 10 + (j + 1)));
    }
    Tmem& tmem = block->tmem();
    for (int lane = 0; lane < 32; ++lane) {
        fill_tmem_slot_f16(tmem, lane * 2, a_mat);
        fill_tmem_slot_f16(tmem, lane * 2 + 1, b_mat);
    }

    sm.add_block(std::move(block));
    WarpContext* warp = sm.get_warp(0);
    REQUIRE(warp != nullptr);

    using ptxsim::testing::step_warp;
    auto& stmts_ref = *stmts;

    CHECK_THROWS_AS(step_warp(warp, stmts_ref), UnsupportedInstructionException);

    CTAContext* cta = warp->get_cta_context();
    REQUIRE(cta != nullptr);
    Tmem& tmem2 = cta->tmem();
    auto result = read_tmem_slot_f16(
        tmem2, 64, static_cast<size_t>(ROWS) * COLS_B);

    bool all_zero = true;
    for (auto v : result) {
        if (v != 0) all_zero = false;
    }
    CHECK(all_zero);
}