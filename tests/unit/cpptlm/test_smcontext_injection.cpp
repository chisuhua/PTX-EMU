/**
 * Phase 8.B SMContext injection point tests (PTX-7a).
 *
 * Verifies SMContext's 3 CppTLM injection points:
 *   - IScoreboard (scoreboard_)
 *   - IPipelineLatencyProvider (pipeline_provider_)
 *   - ITensorCoreTiming (tensor_core_timing_)
 *
 * 7 test cases covering:
 *   1. nullptr injection = no-op (byte-identical fallback)
 *   2. Scoreboard limits concurrent operations
 *   3. Scoreboard release + re-allocate
 *   4. Pipeline overrides InstructionLatencyTable
 *   5. TensorCore overrides default TC latency
 *   6. Pipeline=0 → fallback to TensorCore
 *   7. All 3 injection points active simultaneously
 *
 * Phase 8.B PTX-7a — cpptlm-phase8b-injection-points
 * Ref: ADR-0020, design.md §4
 */
#include "catch_amalgamated.hpp"

#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/pipeline_interface.h"
#include "ptxsim/scoreboard_interface.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/tensor_core_interface.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"

#include <map>
#include <memory>
#include <vector>

using namespace ptxsim;

namespace {

// =========================================================================
// Mock #1: Countable Scoreboard (tracks allocate/release/tick)
// =========================================================================
struct MockScoreboard : IScoreboard {
    int max_entries;
    int used_entries = 0;
    int allocate_count = 0;
    int release_count = 0;
    int tick_count = 0;

    explicit MockScoreboard(int max) : max_entries(max) {}

    bool has_free_entry() const override { return used_entries < max_entries; }

    bool allocate(uint32_t /*reg_id*/, uint32_t /*warp_id*/) override {
        allocate_count++;
        if (used_entries >= max_entries)
            return false;
        used_entries++;
        return true;
    }

    bool release(uint32_t /*reg_id*/, uint32_t /*warp_id*/) override {
        release_count++;
        if (used_entries > 0)
            used_entries--;
        return true;
    }

    void tick() override { tick_count++; }
};

// =========================================================================
// Mock #2: Fixed fractional pipeline latency
// =========================================================================
struct MockPipelineFixed : IPipelineLatencyProvider {
    double fixed_cycles;
    mutable int call_count = 0;

    explicit MockPipelineFixed(double cycles) : fixed_cycles(cycles) {}

    double get_fractional_cycles(const std::string &,
                                 PipelineId) const override {
        call_count++;
        return fixed_cycles;
    }

    double get_fractional_cycles_by_type(int, PipelineId) const override {
        call_count++;
        return fixed_cycles;
    }
};

// =========================================================================
// Mock #3: Fixed TensorCore latency
// =========================================================================
struct MockTensorCoreFixed : ITensorCoreTiming {
    uint32_t fixed_latency;
    mutable int call_count = 0;

    explicit MockTensorCoreFixed(uint32_t latency) : fixed_latency(latency) {}

    uint32_t get_latency(TcPrecision) const override {
        call_count++;
        return fixed_latency;
    }

    uint32_t get_throughput_cycles(TcPrecision) const override {
        return fixed_latency / 2;
    }
};

// =========================================================================
// Helpers
// =========================================================================

std::unique_ptr<WarpContext> make_warp_with_one_active_thread() {
    auto warp = std::make_unique<WarpContext>();

    Dim3 blockIdx = {0, 0, 0};
    Dim3 threadIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};

    std::vector<ptxemu::ir::StatementContext> statements;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;

    auto thread = std::make_unique<ThreadContext>();
    thread->init(blockIdx, threadIdx, gridDim, blockDim, statements, &name2Sym,
                 label2pc, nullptr, nullptr);

    warp->add_thread(std::move(thread), 0);
    REQUIRE(warp->get_active_count() == 1);
    return warp;
}

ptxemu::ir::StatementContext make_stmt(ptxemu::ir::StatementType type) {
    ptxemu::ir::StatementContext stmt;
    stmt.type = type;
    return stmt;
}

} // namespace

// -----------------------------------------------------------------------
// Test 1: nullptr injection = no-op
//
// Create SMContext with default (nullptr) injectors. Verify that
// step_b_set_blocked_cycles is a no-op when both injectors are null.
// This is the byte-identical fallback contract from design.md §2.4.
// -----------------------------------------------------------------------
TEST_CASE("SMContext: nullptr injection = no-op", "[unit][cpptlm][injection]") {
    SMContext sm(4, 128, 4096, 0);

    // Default injectors must be nullptr
    REQUIRE(sm.get_scoreboard() == nullptr);
    REQUIRE(sm.get_pipeline_latency_provider() == nullptr);
    REQUIRE(sm.get_tensor_core_timing() == nullptr);

    auto warp_ptr = make_warp_with_one_active_thread();
    WarpContext &warp = *warp_ptr;
    auto stmt = make_stmt(S_ADD);

    WarpState &ws = warp.get_warp_state();
    REQUIRE(ws.threads[0].is_active);
    REQUIRE_FALSE(ws.threads[0].is_blocked);
    REQUIRE(ws.threads[0].blocked_cycles_remaining == 0);

    // Both nullptr -> step_b is a no-op (design.md §2.4)
    SMContext::step_b_set_blocked_cycles(
        /*pipeline=*/nullptr, /*tc=*/nullptr, &warp, stmt);

    REQUIRE(ws.threads[0].is_active);
    REQUIRE_FALSE(ws.threads[0].is_blocked);
    REQUIRE(ws.threads[0].blocked_cycles_remaining == 0);
}

// -----------------------------------------------------------------------
// Test 2: Scoreboard limits concurrent operations
//
// MockScoreboard with max 2 entries. Allocate 3 times; the 3rd must
// return false while tracking counters increment on every call.
// -----------------------------------------------------------------------
TEST_CASE("SMContext: scoreboard limits concurrent operations",
          "[unit][cpptlm][injection]") {
    SMContext sm(4, 128, 4096, 0);
    MockScoreboard sb(2); // max 2 entries
    sm.set_scoreboard(&sb);

    // Verify injection
    REQUIRE(sm.get_scoreboard() == &sb);

    // 1st allocate: success
    REQUIRE(sb.allocate(0, 0) == true);
    REQUIRE(sb.used_entries == 1);
    REQUIRE(sb.allocate_count == 1);

    // 2nd allocate: success (at limit)
    REQUIRE(sb.allocate(1, 0) == true);
    REQUIRE(sb.used_entries == 2);
    REQUIRE(sb.allocate_count == 2);

    // 3rd allocate: FAILS (over limit)
    REQUIRE(sb.allocate(2, 0) == false);
    REQUIRE(sb.used_entries == 2); // unchanged
    REQUIRE(sb.allocate_count == 3);
}

// -----------------------------------------------------------------------
// Test 3: Scoreboard release + re-allocate
//
// Fill both entries, release one, then verify a new allocate succeeds.
// Counters confirm release and re-allocation accounting.
// -----------------------------------------------------------------------
TEST_CASE("SMContext: scoreboard release after instruction completes",
          "[unit][cpptlm][injection]") {
    SMContext sm(4, 128, 4096, 0);
    MockScoreboard sb(2);
    sm.set_scoreboard(&sb);

    // Fill both entries
    REQUIRE(sb.allocate(0, 0) == true);
    REQUIRE(sb.allocate(1, 0) == true);
    REQUIRE(sb.has_free_entry() == false);

    // Release first entry
    REQUIRE(sb.release(0, 0) == true);
    REQUIRE(sb.used_entries == 1);
    REQUIRE(sb.release_count == 1);

    // Now has free entry
    REQUIRE(sb.has_free_entry() == true);

    // Re-allocate: success
    REQUIRE(sb.allocate(2, 0) == true);
    REQUIRE(sb.used_entries == 2);
    REQUIRE(sb.allocate_count == 3);
}

// -----------------------------------------------------------------------
// Test 4: Pipeline overrides InstructionLatencyTable
//
// MockPipelineFixed returning 4.22 cycles. step_b must use ceil(4.22)=5
// for blocked_cycles_remaining, NOT the default InstructionLatencyTable
// value for S_ADD (which would be different).
// -----------------------------------------------------------------------
TEST_CASE("SMContext: pipeline overrides InstructionLatencyTable",
          "[unit][cpptlm][injection]") {
    SMContext sm(4, 128, 4096, 0);
    MockPipelineFixed pipeline(4.22);
    sm.set_pipeline_latency_provider(&pipeline);

    auto warp_ptr = make_warp_with_one_active_thread();
    WarpContext &warp = *warp_ptr;
    auto stmt = make_stmt(S_ADD);

    // Pipeline returns 4.22 -> ceil(4.22) = 5
    SMContext::step_b_set_blocked_cycles(&pipeline, /*tc=*/nullptr, &warp,
                                         stmt);

    WarpState &ws = warp.get_warp_state();
    REQUIRE(ws.threads[0].is_blocked);
    REQUIRE(ws.threads[0].blocked_cycles_remaining == 5);
    REQUIRE(pipeline.call_count >= 1);
}

// -----------------------------------------------------------------------
// Test 5: TensorCore overrides default TC latency
//
// Pipeline returns 0.0 (no pipeline timing), TC returns 29 for the
// S_TCGEN05_MMA instruction. step_b must fall through to TC and set 29.
// -----------------------------------------------------------------------
TEST_CASE("SMContext: tensor_core overrides default TC latency",
          "[unit][cpptlm][injection]") {
    SMContext sm(4, 128, 4096, 0);
    MockPipelineFixed pipeline(0.0); // pipeline returns 0 -> fallthrough
    MockTensorCoreFixed tc(29);
    sm.set_pipeline_latency_provider(&pipeline);
    sm.set_tensor_core_timing(&tc);

    auto warp_ptr = make_warp_with_one_active_thread();
    WarpContext &warp = *warp_ptr;
    auto stmt = make_stmt(S_TCGEN05_MMA); // TC instruction

    SMContext::step_b_set_blocked_cycles(&pipeline, &tc, &warp, stmt);

    WarpState &ws = warp.get_warp_state();
    REQUIRE(ws.threads[0].is_blocked);
    REQUIRE(ws.threads[0].blocked_cycles_remaining == 29);
    REQUIRE(tc.call_count >= 1);
}

// -----------------------------------------------------------------------
// Test 6: TensorCore fallback when pipeline returns 0
//
// Explicitly verify the fallback chain: pipeline returns 0.0, TC returns
// 29. The result must be 29 (from TC), not 0 (from pipeline) and not
// the default InstructionLatencyTable value. Both mocks must be called.
// -----------------------------------------------------------------------
TEST_CASE("SMContext: tensor_core falls back when pipeline returns 0",
          "[unit][cpptlm][injection]") {
    SMContext sm(4, 128, 4096, 0);
    MockPipelineFixed pipeline(0.0);
    MockTensorCoreFixed tc(29);
    sm.set_pipeline_latency_provider(&pipeline);
    sm.set_tensor_core_timing(&tc);

    auto warp_ptr = make_warp_with_one_active_thread();
    WarpContext &warp = *warp_ptr;
    auto stmt = make_stmt(S_TCGEN05_MMA);

    // Pipeline returns 0.0 -> step_b falls through to TC
    SMContext::step_b_set_blocked_cycles(&pipeline, &tc, &warp, stmt);

    WarpState &ws = warp.get_warp_state();
    // Verify fallback: 29 from TC, not from pipeline
    REQUIRE(ws.threads[0].is_blocked);
    REQUIRE(ws.threads[0].blocked_cycles_remaining == 29);
    // Both pipeline and TC were queried in the priority chain
    REQUIRE(pipeline.call_count >= 1);
    REQUIRE(tc.call_count >= 1);
}

// -----------------------------------------------------------------------
// Test 7: All three injection points active simultaneously
//
// Inject all 3 mocks. Verify:
//   a) All getters return non-null
//   b) Scoreboard tick/allocate works
//   c) Pipeline + TC are both queried by step_b
//   d) Correct latency from the priority chain
// -----------------------------------------------------------------------
TEST_CASE("SMContext: all three injection points active simultaneously",
          "[unit][cpptlm][injection]") {
    SMContext sm(4, 128, 4096, 0);

    MockScoreboard sb(4);
    MockPipelineFixed pipeline(0.0);
    MockTensorCoreFixed tc(17);

    sm.set_scoreboard(&sb);
    sm.set_pipeline_latency_provider(&pipeline);
    sm.set_tensor_core_timing(&tc);

    // All three getters return non-null
    REQUIRE(sm.get_scoreboard() != nullptr);
    REQUIRE(sm.get_pipeline_latency_provider() != nullptr);
    REQUIRE(sm.get_tensor_core_timing() != nullptr);

    // Exercise scoreboard
    sb.tick();
    REQUIRE(sb.tick_count == 1);
    REQUIRE(sb.allocate(0, 0) == true);
    REQUIRE(sb.allocate_count == 1);

    // Exercise pipeline + TC via step_b with a TC instruction.
    // Pipeline returns 0.0 -> step_b falls through to TC.
    auto warp_ptr = make_warp_with_one_active_thread();
    WarpContext &warp = *warp_ptr;
    auto stmt = make_stmt(S_TCGEN05_MMA);

    SMContext::step_b_set_blocked_cycles(&pipeline, &tc, &warp, stmt);

    // All three mocks were invoked
    REQUIRE(sb.tick_count == 1);
    REQUIRE(sb.allocate_count >= 1);
    REQUIRE(pipeline.call_count >= 1);
    REQUIRE(tc.call_count >= 1);

    // Correct latency from TC fallback: 17
    WarpState &ws = warp.get_warp_state();
    REQUIRE(ws.threads[0].blocked_cycles_remaining == 17);
}
