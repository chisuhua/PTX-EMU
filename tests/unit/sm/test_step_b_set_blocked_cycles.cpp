/**
 * PTX-6 TDD: unit tests for SMContext::step_b_set_blocked_cycles (static).
 *
 * Tests the 4-branch priority chain in step_b:
 *   1. Both injectors nullptr -> NO-OP (byte-identical fallback, spec §2.4)
 *   2. pipeline_provider returns positive frac -> use ceil(frac)
 *   3. tc timing + TC instruction -> use tc->get_latency()
 *   4. Fallback to InstructionLatencyTable (getLatency)
 *
 * Why this test exists:
 *   Commit 5b292a91 fixed a regression where the nullptr path was NOT a
 *   no-op (fell through to getLatency + set_blocked_cycles_for_active).
 *   That regression was only caught by unit_simt_integration indirectly.
 *   This test directly locks all 4 branches to prevent future regressions.
 *
 * Ref: ADR-0020, design.md §7.1, commit 5b292a91 (regression fix)
 */
#include "catch_amalgamated.hpp"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/instruction_latency_table.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/pipeline_interface.h"
#include "ptxsim/tensor_core_interface.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/thread_state.h"
#include "ptxsim/sm_context.h"

#include <map>
#include <memory>
#include <vector>

using namespace ptxsim;

namespace {

// Helper: build a WarpContext with a single active thread at lane 0.
// Mirrors pattern in tests/unit/warp/test_warp_blocked_decrement.cpp.
std::unique_ptr<WarpContext> make_warp_with_one_active_thread() {
    auto warp = std::make_unique<WarpContext>();

    Dim3 blockIdx  = {0, 0, 0};
    Dim3 threadIdx = {0, 0, 0};
    Dim3 gridDim   = {1, 1, 1};
    Dim3 blockDim  = {32, 1, 1};

    std::vector<ptxemu::ir::StatementContext> statements;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;

    auto thread = std::make_unique<ThreadContext>();
    thread->init(blockIdx, threadIdx, gridDim, blockDim, statements,
                 &name2Sym, label2pc, nullptr, nullptr);

    warp->add_thread(std::move(thread), 0);
    REQUIRE(warp->get_active_count() == 1);
    return warp;
}

// Helper: build a StatementContext of given type.
ptxemu::ir::StatementContext make_stmt(ptxemu::ir::StatementType type) {
    ptxemu::ir::StatementContext stmt;
    stmt.type = type;
    return stmt;
}

// Mock IPipelineLatencyProvider with configurable return value.
struct ConfigurablePipelineProvider : IPipelineLatencyProvider {
    double return_value = 0.0;

    double get_fractional_cycles(const std::string&,
                                 PipelineId) const override {
        return return_value;
    }
    double get_fractional_cycles_by_type(int, PipelineId) const override {
        return return_value;
    }
};

// Mock ITensorCoreTiming with configurable return value.
struct ConfigurableTensorCoreTiming : ITensorCoreTiming {
    uint32_t return_latency = 10;

    uint32_t get_latency(TcPrecision) const override {
        return return_latency;
    }
    uint32_t get_throughput_cycles(TcPrecision) const override {
        return 5;
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// Branch 1: Both injectors nullptr -> NO-OP
// This is the regression fixed by commit 5b292a91: pre-fix, the nullptr
// path fell through to getLatency + set_blocked_cycles_for_active, which
// was NEW behavior not present in pre-change exe_once().
// ---------------------------------------------------------------------------
TEST_CASE("step_b_set_blocked_cycles: both nullptr = no-op (byte-identical)",
          "[unit][sm][ptx6][step_b][injection]") {
    auto warp_ptr = make_warp_with_one_active_thread();
    WarpContext& warp = *warp_ptr;
    auto stmt = make_stmt(S_ADD);

    WarpState& ws = warp.get_warp_state();
    REQUIRE(ws.threads[0].is_active);
    REQUIRE_FALSE(ws.threads[0].is_blocked);
    REQUIRE(ws.threads[0].blocked_cycles_remaining == 0);

    SMContext::step_b_set_blocked_cycles(
        /*pipeline=*/nullptr, /*tc=*/nullptr, &warp, stmt);

    // NO-OP contract: no state change.
    REQUIRE(ws.threads[0].is_active);
    REQUIRE_FALSE(ws.threads[0].is_blocked);
    REQUIRE(ws.threads[0].blocked_cycles_remaining == 0);
}

// ---------------------------------------------------------------------------
// Branch 2: pipeline_provider returns positive fractional -> ceil(frac)
// ---------------------------------------------------------------------------
TEST_CASE("step_b_set_blocked_cycles: pipeline provider positive frac -> ceil",
          "[unit][sm][ptx6][step_b][injection]") {
    auto warp_ptr = make_warp_with_one_active_thread();
    WarpContext& warp = *warp_ptr;
    auto stmt = make_stmt(S_ADD);

    ConfigurablePipelineProvider pipeline;
    pipeline.return_value = 2.5;  // ceil(2.5) = 3

    SMContext::step_b_set_blocked_cycles(
        &pipeline, /*tc=*/nullptr, &warp, stmt);

    WarpState& ws = warp.get_warp_state();
    REQUIRE(ws.threads[0].is_blocked);
    REQUIRE(ws.threads[0].blocked_cycles_remaining == 3);
}

// ---------------------------------------------------------------------------
// Branch 3: tc timing + TC instruction -> tc->get_latency()
// Pipeline returns 0 so the chain falls through to tc.
// ---------------------------------------------------------------------------
TEST_CASE("step_b_set_blocked_cycles: tc timing + TC instruction -> tc latency",
          "[unit][sm][ptx6][step_b][injection]") {
    auto warp_ptr = make_warp_with_one_active_thread();
    WarpContext& warp = *warp_ptr;
    auto stmt = make_stmt(S_TCGEN05_MMA);

    ConfigurablePipelineProvider pipeline;
    pipeline.return_value = 0.0;  // pipeline returns 0 -> fall to tc

    ConfigurableTensorCoreTiming tc;
    tc.return_latency = 7;

    SMContext::step_b_set_blocked_cycles(&pipeline, &tc, &warp, stmt);

    WarpState& ws = warp.get_warp_state();
    REQUIRE(ws.threads[0].is_blocked);
    REQUIRE(ws.threads[0].blocked_cycles_remaining == 7);
}

// ---------------------------------------------------------------------------
// Branch 4: Fallback to InstructionLatencyTable (getLatency)
// Pipeline returns 0, tc is nullptr (or non-TC instruction), so the chain
// falls to ptxsim::getLatency(stmt.type).cycles.
// ---------------------------------------------------------------------------
TEST_CASE("step_b_set_blocked_cycles: fallback to InstructionLatencyTable",
          "[unit][sm][ptx6][step_b][injection]") {
    auto warp_ptr = make_warp_with_one_active_thread();
    WarpContext& warp = *warp_ptr;
    // S_ADD is non-TC; pipeline returns 0; tc is nullptr -> fallback.
    auto stmt = make_stmt(S_ADD);

    ConfigurablePipelineProvider pipeline;
    pipeline.return_value = 0.0;

    SMContext::step_b_set_blocked_cycles(&pipeline, /*tc=*/nullptr, &warp, stmt);

    WarpState& ws = warp.get_warp_state();
    // Expected: blocked_cycles_remaining == getLatency(S_ADD).cycles
    uint32_t expected = ptxsim::getLatency(S_ADD).cycles;
    REQUIRE(expected > 0);  // sanity: S_ADD must have a non-zero latency
    REQUIRE(ws.threads[0].is_blocked);
    REQUIRE(ws.threads[0].blocked_cycles_remaining == expected);
}
