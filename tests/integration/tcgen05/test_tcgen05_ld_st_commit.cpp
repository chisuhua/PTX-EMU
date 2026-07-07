// tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp
// Phase 2.2 (Fix #13): integration test for tcgen05 ld → mma → commit → wait → st
// roundtrip via the instruction execution pipeline.
//
// TDD RED phase: this test MUST FAIL because tcgen05.commit/wait handlers
// are not yet implemented (stubs throw UnsupportedInstructionException).
//
// Uses ptxsim::testing::step_warp to drive execution (per AGENTS.md
// "禁止在测试代码里重新实现 step_warp").
//
// Gate P1-3.G3: ctest -R "tcgen05_ld_st_commit" PASS

#include "catch_amalgamated.hpp"

#include "ptx_ir/statement_factory.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/memory/tma_descriptor.h"
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

StatementContext make_ld_stmt() {
    using namespace ptxir::factory;
    std::vector<Qualifier> quals;
    quals.push_back(Qualifier::Q_CLUSTER);
    quals.push_back(Qualifier::Q_F16);
    quals.push_back(Qualifier::Q_TCGEN05_LD);
    // B2 factory verification alias. Do not remove until implement-tcgen05-handlers-core.
    auto tcgen05_alias =
        makeTcgen05Instr(Tcgen05OpKind::LD, quals, {}, "tcgen05.ld");
    static_assert(std::is_same_v<decltype(tcgen05_alias), StatementContext>);
    (void)tcgen05_alias;
    return makeWmmaInstr(WmmaType::WMMA_LOAD, quals, {}, "tcgen05.ld");
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

StatementContext make_commit_stmt() {
    using namespace ptxir::factory;
    std::vector<Qualifier> quals;
    quals.push_back(Qualifier::Q_CLUSTER);
    quals.push_back(Qualifier::Q_F16);
    quals.push_back(Qualifier::Q_TCGEN05_COMMIT);
    // B2 factory verification alias. Do not remove until implement-tcgen05-handlers-core.
    auto tcgen05_alias =
        makeTcgen05Instr(Tcgen05OpKind::COMMIT, quals, {}, "tcgen05.commit");
    static_assert(std::is_same_v<decltype(tcgen05_alias), StatementContext>);
    (void)tcgen05_alias;
    return makeWmmaInstr(WmmaType::WMMA_COMMIT, quals, {}, "tcgen05.commit");
}

StatementContext make_wait_stmt() {
    using namespace ptxir::factory;
    std::vector<Qualifier> quals;
    quals.push_back(Qualifier::Q_CLUSTER);
    quals.push_back(Qualifier::Q_F16);
    quals.push_back(Qualifier::Q_TCGEN05_WAIT);
    // B2 factory verification alias. Do not remove until implement-tcgen05-handlers-core.
    auto tcgen05_alias =
        makeTcgen05Instr(Tcgen05OpKind::WAIT, quals, {}, "tcgen05.wait");
    static_assert(std::is_same_v<decltype(tcgen05_alias), StatementContext>);
    (void)tcgen05_alias;
    return makeWmmaInstr(WmmaType::WMMA_WAIT, quals, {}, "tcgen05.wait");
}

StatementContext make_st_stmt() {
    using namespace ptxir::factory;
    std::vector<Qualifier> quals;
    quals.push_back(Qualifier::Q_CLUSTER);
    quals.push_back(Qualifier::Q_F16);
    quals.push_back(Qualifier::Q_TCGEN05_ST);
    // B2 factory verification alias. Do not remove until implement-tcgen05-handlers-core.
    auto tcgen05_alias =
        makeTcgen05Instr(Tcgen05OpKind::ST, quals, {}, "tcgen05.st");
    static_assert(std::is_same_v<decltype(tcgen05_alias), StatementContext>);
    (void)tcgen05_alias;
    return makeWmmaInstr(WmmaType::WMMA_STORE, quals, {}, "tcgen05.st");
}

} // anonymous namespace

TEST_CASE("tcgen05 commit/wait integration: mma→commit→wait→st roundtrip",
          "[integration][tcgen05][commit][wait][roundtrip]") {
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

    // Build instruction pipeline: mma → commit → wait → st → ret
    auto stmts = std::make_shared<std::vector<StatementContext>>();
    stmts->push_back(make_mma_stmt());
    stmts->push_back(make_commit_stmt());
    stmts->push_back(make_wait_stmt());
    stmts->push_back(make_st_stmt());
    stmts->push_back(ptxsim::testing::make_ret());

    auto name2Sym =
        std::make_shared<std::map<std::string, std::unique_ptr<Symtable>>>();

    std::map<std::string, int> label2pc;
    block->init(gridDim, blockDim, blockIdx, *stmts, &*name2Sym, label2pc);

    Tmem& tmem = block->tmem();

    // Prepare TMEM with A and B matrix data for mma
    std::vector<uint16_t> a_mat(ROWS * COLS_A, 0);
    for (int i = 0; i < ROWS; ++i)
        a_mat[i * COLS_A + i] = f32_to_f16(1.0f);

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

    // Set up TMA descriptor for st: points to a destination buffer
    std::array<uint8_t, kTmaDescriptorSize> st_raw{};
    {
        uint64_t gaddr = 0;
        std::memcpy(st_raw.data(), &gaddr, sizeof(gaddr));
        for (int d = 0; d < 5; ++d) st_raw[8 + d * 4] = 8;
        for (int d = 0; d < 5; ++d) st_raw[64 + d * 4] = 1;
        for (int d = 0; d < 4; ++d) {
            st_raw[32 + d * 8] = 0x10;
            st_raw[84 + d] = 1;
        }
        st_raw[88] = 2;
        st_raw[92] = 1; st_raw[93] = 0; st_raw[94] = 0; st_raw[95] = 0;
        st_raw[28] = 0; st_raw[29] = 0; st_raw[30] = 0; st_raw[31] = 0;
    }

    TmaDescriptor st_desc = parse_descriptor_bytes(st_raw.data(), kTmaDescriptorSize);
    std::array<uint8_t, Tmem::kSlotSize> dst_buf{};
    st_desc.global_address = reinterpret_cast<uint64_t>(dst_buf.data());

    sm.add_block(std::move(block));
    WarpContext* warp = sm.get_warp(0);
    REQUIRE(warp != nullptr);

    CTAContext* cta = warp->get_cta_context();
    REQUIRE(cta != nullptr);

    cta->tma_descriptor_store().store(0, st_desc);

    using ptxsim::testing::step_warp;
    auto& stmts_ref = *stmts;

    // Drive mma instruction (pc=0)
    int pc_mma = step_warp(warp, stmts_ref);
    REQUIRE(pc_mma == 0);

    int pc_mma_done = step_warp(warp, stmts_ref);
    REQUIRE(pc_mma_done == 1);

    // Verify mma wrote results to TMEM slots 64-95
    for (int lane = 0; lane < 32; ++lane) {
        auto result = read_tmem_slot_f16(
            cta->tmem(), 64 + static_cast<size_t>(lane),
            static_cast<size_t>(ROWS) * COLS_B);

        REQUIRE(result.size() == static_cast<size_t>(ROWS) * COLS_B);
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS_B; ++j) {
                CAPTURE(lane, i, j);
                REQUIRE(result[i * COLS_B + j] == b_mat[i * COLS_B + j]);
            }
        }
    }

    // Drive commit instruction (pc=2)
    int pc_commit_done = step_warp(warp, stmts_ref);
    REQUIRE(pc_commit_done == 2);

    // Drive wait instruction → should unblock (commit already done)
    int pc_wait_done = step_warp(warp, stmts_ref);
    REQUIRE(pc_wait_done == 3);

    // Drive st instruction (pc=4)
    int pc_st_done = step_warp(warp, stmts_ref);
    REQUIRE(pc_st_done == 4);

    // Verify st wrote data to dst_buf
    const uint16_t* dst_result =
        reinterpret_cast<const uint16_t*>(dst_buf.data());
    // st writes slot 0 (pre-mma A data for lane 0) to dst_buf
    for (size_t i = 0; i < 64; ++i) {
        CAPTURE(i);
        REQUIRE(dst_result[i] == a_mat[i]);
    }

    // Drive ret
    int pc_ret = step_warp(warp, stmts_ref);
    REQUIRE(pc_ret == -1);
}

TEST_CASE("tcgen05 wait blocks warp when no commit issued",
          "[integration][tcgen05][commit][wait][blocking]") {
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
    stmts->push_back(make_commit_stmt());
    stmts->push_back(make_wait_stmt());
    stmts->push_back(ptxsim::testing::make_ret());

    auto name2Sym =
        std::make_shared<std::map<std::string, std::unique_ptr<Symtable>>>();

    std::map<std::string, int> label2pc;
    block->init(gridDim, blockDim, blockIdx, *stmts, &*name2Sym, label2pc);

    // Fill TMEM with test data
    Tmem& tmem = block->tmem();
    std::vector<uint16_t> a_mat(ROWS * COLS_A, 0);
    for (int i = 0; i < ROWS; ++i) a_mat[i * COLS_A + i] = f32_to_f16(1.0f);
    std::vector<uint16_t> b_mat(ROWS * COLS_B, 0);
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < COLS_B; ++j)
            b_mat[i * COLS_B + j] = f32_to_f16(static_cast<float>((i + 1) * 10 + (j + 1)));
    for (int lane = 0; lane < 32; ++lane) {
        fill_tmem_slot_f16(tmem, lane * 2, a_mat);
        fill_tmem_slot_f16(tmem, lane * 2 + 1, b_mat);
    }

    sm.add_block(std::move(block));
    WarpContext* warp = sm.get_warp(0);
    REQUIRE(warp != nullptr);

    using ptxsim::testing::step_warp;
    auto& stmts_ref = *stmts;

    int pc_mma = step_warp(warp, stmts_ref);
    REQUIRE(pc_mma == 0);

    int pc_mma_done = step_warp(warp, stmts_ref);
    REQUIRE(pc_mma_done == 1);

    // commit — advances counter
    int pc_commit = step_warp(warp, stmts_ref);
    REQUIRE(pc_commit == 2);

    // wait — should advance past wait to ret (PC 3)
    int pc_wait = step_warp(warp, stmts_ref);
    REQUIRE(pc_wait == 3);
}

TEST_CASE("tcgen05 mma without commit before wait: commit unblocks wait",
          "[integration][tcgen05][commit][wait][unblock]") {
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
    stmts->push_back(make_wait_stmt());
    stmts->push_back(make_commit_stmt());
    stmts->push_back(ptxsim::testing::make_ret());

    auto name2Sym =
        std::make_shared<std::map<std::string, std::unique_ptr<Symtable>>>();

    std::map<std::string, int> label2pc;
    block->init(gridDim, blockDim, blockIdx, *stmts, &*name2Sym, label2pc);

    Tmem& tmem = block->tmem();
    std::vector<uint16_t> a_mat(ROWS * COLS_A, 0);
    for (int i = 0; i < ROWS; ++i) a_mat[i * COLS_A + i] = f32_to_f16(1.0f);
    std::vector<uint16_t> b_mat(ROWS * COLS_B, 0);
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < COLS_B; ++j)
            b_mat[i * COLS_B + j] = f32_to_f16(static_cast<float>((i + 1) * 10 + (j + 1)));
    for (int lane = 0; lane < 32; ++lane) {
        fill_tmem_slot_f16(tmem, lane * 2, a_mat);
        fill_tmem_slot_f16(tmem, lane * 2 + 1, b_mat);
    }

    sm.add_block(std::move(block));
    WarpContext* warp = sm.get_warp(0);
    REQUIRE(warp != nullptr);

    using ptxsim::testing::step_warp;
    auto& stmts_ref = *stmts;

    int pc_mma = step_warp(warp, stmts_ref);
    REQUIRE(pc_mma == 0);

    int pc_mma_done = step_warp(warp, stmts_ref);
    REQUIRE(pc_mma_done == 1);

    // wait BEFORE commit — warp should be blocked, active_count=0
    int pc_wait = step_warp(warp, stmts_ref);
    REQUIRE(pc_wait == 2);

    // commit — should unblock the warp and advance to ret (pc=3)
    int pc_commit = step_warp(warp, stmts_ref);
    REQUIRE(pc_commit == 3);
}