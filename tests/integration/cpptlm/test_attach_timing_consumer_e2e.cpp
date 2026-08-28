/**
 * @file test_attach_timing_consumer_e2e.cpp
 * @brief PTX-EMU-side reverse-direction e2e tests for IPtxEmuDevice::attach_timing
 *        (HSK-8 spec §CppTLM 端接受条件 #1 + Decision 6 namespace bridge)
 *
 * Purpose
 * -------
 * CppTLM commit d909407 added 7 tests verifying the consumer-side flow
 * (facade.attach_timing -> ScoreboardTLM.allocate/release etc.). These tests
 * close the asymmetric gap by verifying the producer-side flow:
 *   - namespace bridge round-trip identity (G1/G2/G3 partial coverage via
 *     TrackingScoreboard/FixedPipeline/CountingTC being queried post-attach)
 *   - step_a_scoreboard_check / step_c_release_scoreboard query the injected
 *     IScoreboard (G1 + G4)
 *   - step_b_set_blocked_cycles queries IPipelineLatencyProvider (G2 + G4)
 *   - step_b_set_blocked_cycles queries ITensorCoreTiming on TC-instr path
 *     when pipeline returns 0 (G3)
 *
 * All 4 tests go through the PUBLIC `IPtxEmuDevice::attach_timing` API
 * (not direct sm->set_*), so a regression in device_api_impl.cc:299-310
 * (namespace bridge) is caught.
 *
 * Per ptx-lessons-learned §1 (跨模块间接状态翻译): tests drive actual warp
 * setup via WarpExecutorTestFixture, not direct sm->warps[] manipulation.
 *
 * Test naming: [integration][cpptlm][attach_timing] + [g1-g4] sub-tags.
 *   - G1: scoreboard queried by step_a/c (exe_once path)
 *   - G2: pipeline queried by step_b (S_FMA path)
 *   - G3: tensor_core queried by step_b (S_TCGEN05_MMA path)
 *   - G4: e2e — exe_once queries all 3 injected interfaces
 */

#include "catch_amalgamated.hpp"
#include "ptxemu/testing/warp_executor_test_fixture.h"

#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/gpu_context.h"
#include "ptxsim/pipeline_interface.h"
#include "ptxsim/scoreboard_interface.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/tensor_core_interface.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/warp_context.h"

#include <memory>

using ptxemu::IPtxEmuDevice;
using ptxemu::testing::WarpExecutorTestFixture;
// StatementContext / StatementType / GenericInstr / Qualifier are in
// global namespace (from include/ptx_ir/, no namespace wrap)

namespace {

// ============================================================================
// Mock: IScoreboard that counts allocate/release calls
// ============================================================================
struct TrackingScoreboard : ::IScoreboard {
    int alloc_calls = 0;
    int release_calls = 0;
    int tick_calls = 0;

    bool has_free_entry() const override {
        return true;  // never block scheduling
    }
    bool allocate(uint32_t /*reg_id*/, uint32_t /*warp_id*/) override {
        ++alloc_calls;
        return true;
    }
    bool release(uint32_t /*reg_id*/, uint32_t /*warp_id*/) override {
        ++release_calls;
        return true;
    }
    void tick() override {
        ++tick_calls;
    }
};

// ============================================================================
// Mock: IPipelineLatencyProvider that counts get_fractional_cycles_by_type
// ============================================================================
struct FixedPipeline : ::IPipelineLatencyProvider {
    double fractional_cycles = 4.22;
    mutable int cycles_calls = 0;

    double get_fractional_cycles(const std::string&,
                                 PipelineId) const override {
        ++cycles_calls;
        return fractional_cycles;
    }
    double get_fractional_cycles_by_type(int,
                                         PipelineId) const override {
        ++cycles_calls;
        return fractional_cycles;
    }
};

// ============================================================================
// Mock: ITensorCoreTiming that counts get_latency calls
// ============================================================================
struct CountingTC : ::ITensorCoreTiming {
    mutable int latency_calls = 0;
    mutable TcPrecision last_precision = TcPrecision::FP16;

    uint32_t get_latency(TcPrecision prec) const override {
        ++latency_calls;
        last_precision = prec;
        return 8;
    }
    uint32_t get_throughput_cycles(TcPrecision) const override {
        return 1;
    }
};

// ============================================================================
// Helper: attach mocks through IPtxEmuDevice public API.
// Performs the HSK-8 spec Decision6 namespace bridge (static_cast<void*>
// round-trip) and calls device->attach_timing(...).
// ============================================================================
void attach_through_device(IPtxEmuDevice* dev, ::IScoreboard* sb,
                           ::IPipelineLatencyProvider* pl,
                           ::ITensorCoreTiming* tc) {
    REQUIRE(dev != nullptr);

    // Forward-declared types (ptxemu::IScoreboard etc.) come from device_api.h:36-38.
    // Round-trip: ::IScoreboard* -> void* -> ptxemu::IScoreboard*
    ptxemu::IScoreboard* p_sb = nullptr;
    ptxemu::IPipelineLatencyProvider* p_pl = nullptr;
    ptxemu::ITensorCoreTiming* p_tc = nullptr;
    if (sb) {
        p_sb = static_cast<ptxemu::IScoreboard*>(static_cast<void*>(sb));
    }
    if (pl) {
        p_pl = static_cast<ptxemu::IPipelineLatencyProvider*>(
            static_cast<void*>(pl));
    }
    if (tc) {
        p_tc = static_cast<ptxemu::ITensorCoreTiming*>(static_cast<void*>(tc));
    }

    dev->attach_timing(p_sb, p_pl, p_tc);
}

// Helper for constructing a generic S_TCGEN05_MMA statement (for G3).
// make_stmt(S_TCGEN05_MMA) only sets stmt.type, but step_b falls into the
// TC branch when is_tensor_core_instruction returns true (which checks
// stmt.type range only — no operand dependency). The default precision
// (FP16) from map_instruction_to_tc_precision is acceptable for verifying
// get_latency was called.
ptxemu::ir::StatementContext make_tcgen05_mma_stmt() {
    ptxemu::ir::StatementContext stmt;
    stmt.type = ptxemu::ir::StatementType::S_TCGEN05_MMA;
    // Empty operands is fine: tc path only checks stmt.type range
    return stmt;
}

}  // namespace

// ============================================================================
// G1: scoreboard queried by exe_once step_a/c via attach_timing
// ============================================================================
//
// Inject TrackingScoreboard via IPtxEmuDevice::attach_timing, then run
// sm_exe_once() with one warp executing one schedulable S_FMA. The warp
// must carry the S_FMA statement (via WarpExecutorTestFixture's new
// optional statements parameter) so step_a_scoreboard_check /
// step_c_release_scoreboard can extract a non-empty dest_regs vector and
// actually call sb.allocate / sb.release.
TEST_CASE("attach_timing: scoreboard queried by exe_once step_a/c",
          "[integration][cpptlm][attach_timing][g1]") {
    WarpExecutorTestFixture fix({ptxsim::testing::make_ffma(
        "%f0", "%f1", "%f2", "%f3")});

    TrackingScoreboard sb;
    attach_through_device(fix.dev(), &sb, nullptr, nullptr);

    // Verify round-trip identity (namespace bridge sanity)
    REQUIRE(fix.sm()->get_scoreboard() == &sb);

    // Drive exe_once via IPtxEmuDevice public API.
    int rc = fix.dev()->sm_exe_once(0);
    REQUIRE((rc == 0 || rc == -1));  // -1 acceptable: SM/scheduler idle skip

    // step_a_scoreboard_check (sm_context.cpp:273) + step_c_release_scoreboard
    // (sm_context.cpp:316) must call the injected sb.allocate / sb.release.
    REQUIRE(sb.alloc_calls > 0);
    REQUIRE(sb.release_calls > 0);
}

// ============================================================================
// G2: pipeline queried by step_b (S_FMA) via attach_timing
// ============================================================================
//
// Inject FixedPipeline via IPtxEmuDevice::attach_timing, then call
// SMContext::step_b_set_blocked_cycles (public static) directly with a
// non-TC S_FMA statement. The injected pipeline's
// get_fractional_cycles_by_type must be invoked.
TEST_CASE("attach_timing: pipeline queried by step_b (S_FMA)",
          "[integration][cpptlm][attach_timing][g2]") {
    WarpExecutorTestFixture fix;  // default empty statements OK for G2

    FixedPipeline pipeline;
    attach_through_device(fix.dev(), nullptr, &pipeline, nullptr);

    REQUIRE(fix.sm()->get_pipeline_latency_provider() == &pipeline);

    auto stmt = ptxsim::testing::make_ffma("%f0", "%f1", "%f2", "%f3");
    SMContext::step_b_set_blocked_cycles(
        fix.sm()->get_pipeline_latency_provider(), /*tc=*/nullptr, fix.warp(),
        stmt);

    REQUIRE(pipeline.cycles_calls > 0);
}

// ============================================================================
// G3: tensor_core queried by step_b (S_TCGEN05_MMA, pipeline returns 0)
// ============================================================================
//
// Inject CountingTC via IPtxEmuDevice::attach_timing alongside a pipeline
// that returns 0 fractional_cycles. step_b_set_blocked_cycles then falls
// into the TC fallback path (sm_context_cpptlm_inject.cpp:28-32) and calls
// tc.get_latency.
TEST_CASE("attach_timing: tensor_core queried by step_b (S_TCGEN05_MMA)",
          "[integration][cpptlm][attach_timing][g3]") {
    WarpExecutorTestFixture fix;

    FixedPipeline pipeline_zero;  // default 4.22; override below
    pipeline_zero.fractional_cycles = 0.0;  // force TC branch
    CountingTC tc;
    attach_through_device(fix.dev(), nullptr, &pipeline_zero, &tc);

    REQUIRE(fix.sm()->get_tensor_core_timing() == &tc);

    auto stmt = make_tcgen05_mma_stmt();
    SMContext::step_b_set_blocked_cycles(
        fix.sm()->get_pipeline_latency_provider(), fix.sm()->get_tensor_core_timing(),
        fix.warp(), stmt);

    REQUIRE(tc.latency_calls > 0);
}

// ============================================================================
// G4: e2e — exe_once queries all 3 injected interfaces
// ============================================================================
//
// Inject all 3 mocks via IPtxEmuDevice::attach_timing, run sm_exe_once()
// with one warp executing one S_FMA, then assert:
//   - scoreboard.allocate_calls > 0 AND scoreboard.release_calls > 0
//     (step_a + step_c both queried)
//   - pipeline.cycles_calls > 0 (step_b pipeline path queried)
//   - tensor_core.latency_calls == 0 (step_b TC path NOT triggered —
//     S_FMA is non-TC, pipeline path takes priority per
//     sm_context_cpptlm_inject.cpp:21-32)
TEST_CASE("attach_timing: e2e — exe_once queries all 3 injected interfaces",
          "[integration][cpptlm][attach_timing][g4][e2e]") {
    WarpExecutorTestFixture fix({ptxsim::testing::make_ffma(
        "%f0", "%f1", "%f2", "%f3")});

    TrackingScoreboard sb;
    FixedPipeline pipeline;
    CountingTC tc;
    attach_through_device(fix.dev(), &sb, &pipeline, &tc);

    REQUIRE(fix.sm()->get_scoreboard() == &sb);
    REQUIRE(fix.sm()->get_pipeline_latency_provider() == &pipeline);
    REQUIRE(fix.sm()->get_tensor_core_timing() == &tc);

    int rc = fix.dev()->sm_exe_once(0);
    REQUIRE((rc == 0 || rc == -1));

    // Step a + c queried the scoreboard
    REQUIRE(sb.alloc_calls > 0);
    REQUIRE(sb.release_calls > 0);

    // Step b queried the pipeline (S_FMA -> P0_INT_FP32, non-TC path)
    REQUIRE(pipeline.cycles_calls > 0);

    // Step b did NOT query the TC (pipeline path takes priority for non-TC
    // instructions; this verifies the priority chain in
    // sm_context_cpptlm_inject.cpp:21-32)
    REQUIRE(tc.latency_calls == 0);
}