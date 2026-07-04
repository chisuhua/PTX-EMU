// test_tcgen05_mma_fragment.cpp
// Phase 1.1 (Fix #10): unit test for tcgen05.mma fragment arithmetic.
//
// TDD RED phase: this test MUST FAIL because WmmaHandler unconditionally
// throws UnsupportedInstructionException. After implementation, it MUST
// PASS by verifying TMEM writes from the fragment arithmetic.
//
// Verifies:
//   1. WmmaHandler no longer throws for tcgen05.mma.cta_group::1.kind::f16
//   2. Correct 8x4 fragment arithmetic per lane
//   3. 32 lanes produce correct output fragment elements

#include "catch_amalgamated.hpp"

#include "ptxsim/instruction_handlers.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/utils/half_utils.h"

#include <array>
#include <cstring>
#include <vector>

namespace {

// Compute C = A * B for f16 values (standard matmul).
// A: rows_a x cols_a, B: cols_a x cols_b (shared inner dim).
void compute_ref_matmul(const std::vector<uint16_t>& a,
                        const std::vector<uint16_t>& b,
                        std::vector<uint16_t>& c,
                        int rows_a, int cols_a, int cols_b) {
    c.resize(static_cast<size_t>(rows_a) * cols_b);
    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < cols_a; ++k) {
                sum += f16_to_f32(a[i * cols_a + k]) *
                       f16_to_f32(b[k * cols_b + j]);
            }
            c[i * cols_b + j] = f32_to_f16(sum);
        }
    }
}

// Fill TMEM slot with 128 bytes of data.
// Each slot holds up to 64 f16 values.
void fill_tmem_slot_with_f16(Tmem& tmem, size_t slot_id,
                              const std::vector<uint16_t>& data) {
    // 128 bytes per slot per PTX ISA §9.7.13
    std::array<uint8_t, Tmem::kSlotSize> buf{};
    size_t copy_bytes =
        std::min(data.size() * sizeof(uint16_t), Tmem::kSlotSize);
    std::memcpy(buf.data(), data.data(), copy_bytes);
    tmem.write(slot_id, buf.data(), Tmem::kSlotSize);
}

// Read f16 values from a TMEM slot.
std::vector<uint16_t> read_tmem_slot_f16(Tmem& tmem, size_t slot_id,
                                          size_t num_values) {
    std::array<uint8_t, Tmem::kSlotSize> buf{};
    tmem.read(slot_id, buf.data(), Tmem::kSlotSize);
    size_t n = std::min(num_values, static_cast<size_t>(Tmem::kSlotSize / 2));
    std::vector<uint16_t> result(n);
    std::memcpy(result.data(), buf.data(), n * sizeof(uint16_t));
    return result;
}

// Build qualifier list matching tcgen05.mma.cta_group::1.kind::f16
std::vector<Qualifier> make_tcgen05_mma_f16_quals() {
    // The exact qualifiers used by PTX parser for
    // tcgen05.mma.cta_group::1.kind::f16
    // We need at minimum Q_F16 to signal f16 data type.
    // The cta_group::1 and kind::f16 parsing may add Q_CLUSTER / Q_CTA
    // qualifiers depending on the parser.
    // For unit-test phase, we pass Q_CLUSTER and Q_F16.
    return {Qualifier::Q_CLUSTER, Qualifier::Q_F16};
}

} // anonymous namespace

TEST_CASE("tcgen05.mma handler throws for bare ThreadContext (no warp)",
          "[unit][ptx][wmma][tcgen05][fragment]") {
    ThreadContext ctx;
    WmmaHandler handler;
    void* ops[4] = {nullptr, nullptr, nullptr, nullptr};
    auto quals = make_tcgen05_mma_f16_quals();

    REQUIRE_THROWS_AS(handler.processWmmaOperation(&ctx, ops, quals),
                      UnsupportedInstructionException);
}

TEST_CASE("tcgen05.mma writes 8x4 f16 fragment per lane to TMEM",
          "[unit][ptx][wmma][tcgen05][fragment]") {
    constexpr int ROWS = 8;
    constexpr int COLS_A = 8;
    constexpr int COLS_B = 4;

    std::vector<uint16_t> a_mat(ROWS * COLS_A, 0);
    for (int i = 0; i < ROWS; ++i) {
        a_mat[i * COLS_A + i] = f32_to_f16(1.0f);
    }

    std::vector<uint16_t> b_mat(ROWS * COLS_B, 0);
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS_B; ++j) {
            float val = static_cast<float>((i + 1) * 10 + (j + 1));
            b_mat[i * COLS_B + j] = f32_to_f16(val);
        }
    }

    CTAContext cta;
    Tmem& tmem = cta.tmem();

    for (int lane = 0; lane < 32; ++lane) {
        fill_tmem_slot_with_f16(tmem, static_cast<size_t>(lane) * 2, a_mat);
        fill_tmem_slot_with_f16(tmem, static_cast<size_t>(lane) * 2 + 1, b_mat);
    }

    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};
    std::vector<StatementContext> stmts;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    cta.init(gridDim, blockDim, blockIdx, stmts, &name2Sym, label2pc);

    auto warps = cta.release_warps();
    REQUIRE(warps.size() == 1);
    WarpContext* warp = warps[0].get();
    REQUIRE(warp != nullptr);

    ThreadContext* ctx = warp->get_thread(0);
    REQUIRE(ctx != nullptr);

    WmmaHandler handler;
    void* ops[4] = {nullptr, nullptr, nullptr, nullptr};
    auto quals = make_tcgen05_mma_f16_quals();

    REQUIRE_NOTHROW(handler.processWmmaOperation(ctx, ops, quals));

    for (int lane = 0; lane < 32; ++lane) {
        auto result = read_tmem_slot_f16(
            tmem, static_cast<size_t>(64 + lane),
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

TEST_CASE("tcgen05.mma handler still throws for non-tcgen05 qualifiers",
          "[unit][ptx][wmma][tcgen05][fragment]") {
    ThreadContext ctx;
    WmmaHandler handler;
    void* ops[4] = {nullptr, nullptr, nullptr, nullptr};

    SECTION("empty qualifiers") {
        std::vector<Qualifier> quals;
        REQUIRE_THROWS_AS(
            handler.processWmmaOperation(&ctx, ops, quals),
            UnsupportedInstructionException);
    }

    SECTION("plain F16 qualifiers (no cta_group)") {
        std::vector<Qualifier> quals;
        quals.push_back(Qualifier::Q_F16);
        quals.push_back(Qualifier::Q_F16);
        REQUIRE_THROWS_AS(
            handler.processWmmaOperation(&ctx, ops, quals),
            UnsupportedInstructionException);
    }
}

TEST_CASE("tcgen05.mma 32-lane fragment element correctness (identity matmul)",
          "[unit][ptx][wmma][tcgen05][fragment][32lane]") {
    constexpr int ROWS = 8;
    constexpr int COLS_A = 8;
    constexpr int COLS_B = 4;

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

    CTAContext cta;
    Tmem& tmem = cta.tmem();

    for (int lane = 0; lane < 32; ++lane) {
        fill_tmem_slot_with_f16(tmem, static_cast<size_t>(lane) * 2, a_mat);
        fill_tmem_slot_with_f16(tmem, static_cast<size_t>(lane) * 2 + 1, b_mat);
    }

    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};
    std::vector<StatementContext> stmts;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    cta.init(gridDim, blockDim, blockIdx, stmts, &name2Sym, label2pc);

    auto warps = cta.release_warps();
    REQUIRE(warps.size() == 1);
    WarpContext* warp = warps[0].get();

    WmmaHandler handler;
    void* ops[4] = {nullptr, nullptr, nullptr, nullptr};
    auto quals = make_tcgen05_mma_f16_quals();

    ThreadContext* ctx = warp->get_thread(0);
    REQUIRE(ctx != nullptr);
    REQUIRE_NOTHROW(handler.processWmmaOperation(ctx, ops, quals));

    for (int lane = 0; lane < 32; ++lane) {
        auto result = read_tmem_slot_f16(
            tmem, static_cast<size_t>(64 + lane),
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

TEST_CASE("tcgen05.mma per-lane (lane_idx -> (row,col)) mapping verified",
          "[unit][ptx][wmma][tcgen05][fragment][mapping]") {
    constexpr int ROWS = 8;
    constexpr int COLS_A = 8;
    constexpr int COLS_B = 4;

    CTAContext cta;
    Tmem& tmem = cta.tmem();

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

        fill_tmem_slot_with_f16(tmem,
                                 static_cast<size_t>(lane) * 2, a_mat);
        fill_tmem_slot_with_f16(tmem,
                                 static_cast<size_t>(lane) * 2 + 1, b_mat);
    }

    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};
    std::vector<StatementContext> stmts;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    cta.init(gridDim, blockDim, blockIdx, stmts, &name2Sym, label2pc);

    auto warps = cta.release_warps();
    REQUIRE(warps.size() == 1);
    WarpContext* warp = warps[0].get();

    WmmaHandler handler;
    void* ops[4] = {nullptr, nullptr, nullptr, nullptr};
    auto quals = make_tcgen05_mma_f16_quals();

    ThreadContext* ctx = warp->get_thread(0);
    REQUIRE(ctx != nullptr);
    REQUIRE_NOTHROW(handler.processWmmaOperation(ctx, ops, quals));

    for (int lane = 0; lane < 32; ++lane) {
        auto result = read_tmem_slot_f16(
            tmem, static_cast<size_t>(64 + lane),
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