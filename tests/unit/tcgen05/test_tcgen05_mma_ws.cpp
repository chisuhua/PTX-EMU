// tests/unit/tcgen05/test_tcgen05_mma_ws.cpp
// =============================================================================
// Phase 3 of implement-tcgen05-handlers-extended (Oracle 2026-07-08 A-path):
// unit tests for the tcgen05.mma.ws qualifier-based routing inside
// processTcgen05Mma.
//
// Per Oracle 2026-07-08 critical findings, Tcgen05OpKind::MMA_WS dispatch
// is unreachable from real PTX (grammar treats .ws as Q_TCGEN_WS qualifier,
// not as sub-op). The ws variant is routed INSIDE processTcgen05Mma via
// a qualifier scan. Q3-A scope: only Q_F16 supported on the ws path.
//
// Tests:
//   1. ws + Q_F16 → no throw (ws path calls helper)
//   2. ws + Q_F32 → throws (Q3-A scope violation)
//   3. ws + no kind → throws (Q3-A scope violation)
//   4. no ws → no throw (regular mma path, no Q3-A check applies)
//
// Note: these tests construct Tcgen05Instr directly (no real TMEM/WarpContext
// setup needed for the ws scope checks — those run BEFORE the helper call).
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instructions/tcgen05.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"

#include <memory>
#include <vector>

namespace {

// Minimal TestRig mirroring tests/unit/tcgen05/test_tcgen05_cp.cpp.
// Provides a fully-linked CTAContext so processTcgen05Mma's cta_context
// check passes before the ws qualifier scope check runs.
class TestRig {
public:
    TestRig()
        : sm_(std::make_unique<SMContext>(/*num_warps=*/1, /*warp_size=*/32,
                                          /*max_ctas=*/1, /*shared_mem=*/4096)),
          cta_(std::make_unique<CTAContext>()),
          warp_(std::make_unique<WarpContext>()),
          thread_(std::make_unique<ThreadContext>()) {
        warp_->set_warp_id(0);
        warp_->set_cta_context(cta_.get());
        thread_->set_warp_context(warp_.get());
    }

    CTAContext &cta() { return *cta_; }
    WarpContext &warp() { return *warp_; }
    ThreadContext &thread() { return *thread_; }

private:
    std::unique_ptr<SMContext> sm_;
    std::unique_ptr<CTAContext> cta_;
    std::unique_ptr<WarpContext> warp_;
    std::unique_ptr<ThreadContext> thread_;
};

// Build a Tcgen05Instr with the given qualifier set and op_kind.
// Default op_kind is MMA (the path real PTX takes via the grammar).
Tcgen05Instr make_instr(Tcgen05OpKind op_kind,
                        std::vector<Qualifier> qualifiers) {
    Tcgen05Instr instr;
    instr.op_kind = op_kind;
    instr.qualifiers = std::move(qualifiers);
    // 4 operands (MMA / MMA_WS have operand count 4 per ptx_op.def:133-134).
    // Operand content is irrelevant — processTcgen05Mma discards `instr`
    // for the arithmetic call (it only reads `instr.qualifiers` for the
    // ws scope check, then calls the helper which reads from TMEM).
    instr.operands = std::vector<OperandContext>(
        4, OperandContext(RegOperand{"r", 0}));
    return instr;
}

}  // namespace

// =============================================================================
// Q3-A happy path: ws + Q_F16 → ws path executes (no throw)
// =============================================================================

TEST_CASE("processTcgen05Mma with Q_TCGEN_WS + Q_F16 executes ws path",
          "[unit][tcgen05][mma_ws][scope][happy_path]") {
    TestRig rig;
    Tcgen05Instr instr = make_instr(
        Tcgen05OpKind::MMA,
        {Qualifier::Q_TCGEN_WS, Qualifier::Q_F16});

    // TMEM is default-zero; the helper will multiply zeros and write zeros
    // back to slot 64..95. No exception expected.
    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), instr));
}

TEST_CASE("processTcgen05Mma with op_kind=MMA_WS w/o Q_TCGEN_WS falls through to regular mma path",
          "[unit][tcgen05][mma_ws][scope][op_kind_mma_ws]") {
    // Tcgen05OpKind::MMA_WS op_kind alone (no Q_TCGEN_WS qualifier)
    // routes to processTcgen05Mma which scans qualifiers; without
    // Q_TCGEN_WS the regular path executes (helper called).
    TestRig rig;
    Tcgen05Instr instr = make_instr(
        Tcgen05OpKind::MMA_WS,
        {Qualifier::Q_F16});

    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), instr));
}

TEST_CASE("processTcgen05Mma with op_kind=MMA_WS + Q_TCGEN_WS + Q_F16 executes ws path via dispatch",
          "[unit][tcgen05][mma_ws][scope][op_kind_mma_ws][dispatch]") {
    // Stresses the dispatch path: op_kind=MMA_WS + Q_TCGEN_WS + Q_F16
    // should execute the ws path (Q3-A scope satisfied).
    TestRig rig;
    Tcgen05Instr instr = make_instr(
        Tcgen05OpKind::MMA_WS,
        {Qualifier::Q_TCGEN_WS, Qualifier::Q_F16});

    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), instr));
}

TEST_CASE("processTcgen05Mma with op_kind=MMA_WS + Q_TCGEN_WS but no Q_F16 throws Q3-A scope",
          "[unit][tcgen05][mma_ws][scope][op_kind_mma_ws][scope_violation]") {
    // Verifies dispatch path also enforces Q3-A scope check.
    TestRig rig;
    Tcgen05Instr instr = make_instr(
        Tcgen05OpKind::MMA_WS,
        {Qualifier::Q_TCGEN_WS});  // no Q_F16

    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Mma(&rig.thread(), instr),
        UnsupportedInstructionException);
}

// =============================================================================
// Q3-A scope violations: ws + non-f16 kind throws
// =============================================================================

TEST_CASE("processTcgen05Mma with Q_TCGEN_WS + Q_F32 throws (Q3-A scope)",
          "[unit][tcgen05][mma_ws][scope][violation][f32]") {
    TestRig rig;
    Tcgen05Instr instr = make_instr(
        Tcgen05OpKind::MMA,
        {Qualifier::Q_TCGEN_WS, Qualifier::Q_F32});

    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Mma(&rig.thread(), instr),
        UnsupportedInstructionException);
}

TEST_CASE("processTcgen05Mma with Q_TCGEN_WS + Q_BF16 throws (Q3-A scope)",
          "[unit][tcgen05][mma_ws][scope][violation][bf16]") {
    TestRig rig;
    Tcgen05Instr instr = make_instr(
        Tcgen05OpKind::MMA,
        {Qualifier::Q_TCGEN_WS, Qualifier::Q_BF16});

    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Mma(&rig.thread(), instr),
        UnsupportedInstructionException);
}

TEST_CASE("processTcgen05Mma with Q_TCGEN_WS + no kind throws (Q3-A scope)",
          "[unit][tcgen05][mma_ws][scope][violation][no_kind]") {
    TestRig rig;
    Tcgen05Instr instr = make_instr(
        Tcgen05OpKind::MMA,
        {Qualifier::Q_TCGEN_WS});  // no kind qualifier

    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Mma(&rig.thread(), instr),
        UnsupportedInstructionException);
}

// =============================================================================
// Negative control: no ws → regular mma (no Q3-A check applies)
// =============================================================================

TEST_CASE("processTcgen05Mma without Q_TCGEN_WS executes regular path",
          "[unit][tcgen05][mma_ws][scope][negative_control]") {
    TestRig rig;
    Tcgen05Instr instr = make_instr(
        Tcgen05OpKind::MMA,
        {Qualifier::Q_F16});  // no Q_TCGEN_WS

    // Regular mma path: no Q3-A scope check, helper called.
    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), instr));
}

TEST_CASE("processTcgen05Mma with op_kind=MMA and empty qualifiers executes regular path",
          "[unit][tcgen05][mma_ws][scope][empty_quals]") {
    TestRig rig;
    Tcgen05Instr instr = make_instr(Tcgen05OpKind::MMA, {});

    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), instr));
}