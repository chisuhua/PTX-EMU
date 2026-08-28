// tests/unit/ptx_ir/test_tcgen05_extended_opkind.cpp
// Phase 4 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q5-C / Q6-B):
// Unit tests for the 6 tcgen05.* extended handlers (alloc/dealloc/relinquish/
// cp/mma.ws/fence) via direct processTcgen05* invocation on a minimal rig.
//
// This file is the unit-level companion to:
//   - tests/integration/tcgen05/test_tcgen05_extended_parse.cpp (step_warp pipeline)
//   - tests/e2e/kernel/test_tcgen05_alloc.cu (real CUDA kernel)
//
// UNVERIFIED-AGAINST-HARDWARE — Oracle Q5-C hand-computed golden for no-op
// fence (records position but causes no observable state mutation outside
// WarpState::fence_position).

#include "catch_amalgamated.hpp"

#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/instructions/tcgen05.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"

#include <memory>
#include <stdexcept>
#include <vector>

namespace {

// Factory initializer — required for InstructionFactory::initialize() to set up
// the dispatch table. Mirrors the pattern from
// tests/integration/tcgen05/test_alloc_dealloc_relinquish.cpp:25-31.
static void init_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

// Minimal rig: WarpContext + ThreadContext. fence is a record-only no-op so
// CTAContext is not strictly required, but we wire it for parity with alloc/cp
// tests and to keep the rig reusable if future test cases touch tmem.
class FenceUnitRig {
public:
    FenceUnitRig() {
        init_factory_once();
        cta_ = std::make_unique<CTAContext>();
        warp_ = std::make_unique<WarpContext>();
        thread_ = std::make_unique<ThreadContext>();

        warp_->set_warp_id(0);
        warp_->set_cta_context(cta_.get());
        thread_->set_warp_context(warp_.get());
    }

    WarpContext& warp() { return *warp_; }
    ThreadContext& thread() { return *thread_; }

private:
    std::unique_ptr<CTAContext> cta_;
    std::unique_ptr<WarpContext> warp_;
    std::unique_ptr<ThreadContext> thread_;
};

ptxemu::ir::Tcgen05Instr make_fence_instr(uint32_t cta_group,
                              std::vector<ptxemu::ir::Qualifier> qualifiers) {
    ptxemu::ir::Tcgen05Instr instr;
    instr.op_kind = ptxemu::ir::Tcgen05OpKind::FENCE;
    instr.cta_group = cta_group;
    instr.qualifiers = std::move(qualifiers);
    return instr;
}

}  // namespace

// -----------------------------------------------------------------------------
// processTcgen05Fence: forward path (before / after thread_sync)
// -----------------------------------------------------------------------------

TEST_CASE("processTcgen05Fence: ::before_thread_sync records kFenceBefore",
          "[unit][tcgen05][fence][handler]") {
    FenceUnitRig rig;

    const bool before_permit = rig.warp().get_allocate_permit();
    const uint32_t before_exec = rig.warp().get_exec_mask();

    REQUIRE_NOTHROW(ptxsim::processTcgen05Fence(
        &rig.thread(),
        make_fence_instr(1, {ptxemu::ir::Qualifier::Q_BEFORE_THREAD_SYNC})));

    REQUIRE(rig.warp().get_last_fence_position() ==
            WarpContext::kFenceBefore);
    REQUIRE(rig.warp().get_allocate_permit() == before_permit);  // invariant
    REQUIRE(rig.warp().get_exec_mask() == before_exec);         // invariant
}

TEST_CASE("processTcgen05Fence: ::after_thread_sync records kFenceAfter",
          "[unit][tcgen05][fence][handler]") {
    FenceUnitRig rig;

    REQUIRE_NOTHROW(ptxsim::processTcgen05Fence(
        &rig.thread(),
        make_fence_instr(1, {ptxemu::ir::Qualifier::Q_AFTER_THREAD_SYNC})));

    REQUIRE(rig.warp().get_last_fence_position() ==
            WarpContext::kFenceAfter);
}

// -----------------------------------------------------------------------------
// processTcgen05Fence: error paths
// -----------------------------------------------------------------------------

TEST_CASE("processTcgen05Fence: cta_group::2 throws ADR-0018",
          "[unit][tcgen05][fence][handler][error][cta_group_2]") {
    FenceUnitRig rig;

    REQUIRE(rig.warp().get_last_fence_position() == WarpContext::kFenceNone);
    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Fence(
            &rig.thread(),
            make_fence_instr(2, {ptxemu::ir::Qualifier::Q_BEFORE_THREAD_SYNC})),
        UnsupportedInstructionException);
    // Fence state must NOT have been mutated before the throw.
    REQUIRE(rig.warp().get_last_fence_position() == WarpContext::kFenceNone);
}

TEST_CASE("processTcgen05Fence: no qualifier throws (sanity for hand-built instr)",
          "[unit][tcgen05][fence][handler][error]") {
    FenceUnitRig rig;
    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Fence(
            &rig.thread(),
            make_fence_instr(1, /*empty*/ {})),
        UnsupportedInstructionException);
}

TEST_CASE("processTcgen05Fence: both qualifiers throws (PTX §9.7.16 violation)",
          "[unit][tcgen05][fence][handler][error]") {
    FenceUnitRig rig;
    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Fence(
            &rig.thread(),
            make_fence_instr(1,
                             {ptxemu::ir::Qualifier::Q_BEFORE_THREAD_SYNC,
                              ptxemu::ir::Qualifier::Q_AFTER_THREAD_SYNC})),
        UnsupportedInstructionException);
}

// -----------------------------------------------------------------------------
// processTcgen05Fence: state-modification audit (Oracle Q5-C invariant suite)
// Verifies fence is a pure no-op marker — only WarpState::fence_position moves.
// -----------------------------------------------------------------------------

TEST_CASE("processTcgen05Fence: state-modification audit (no-mutation invariants)",
          "[unit][tcgen05][fence][handler][audit]") {
    FenceUnitRig rig;

    const bool p = rig.warp().get_allocate_permit();
    const uint32_t m = rig.warp().get_exec_mask();
    const int c = rig.warp().get_active_count();

    ptxsim::processTcgen05Fence(
        &rig.thread(),
        make_fence_instr(1, {ptxemu::ir::Qualifier::Q_BEFORE_THREAD_SYNC}));

    REQUIRE(rig.warp().get_allocate_permit() == p);  // no set_allocate_permit
    REQUIRE(rig.warp().get_exec_mask() == m);         // no set_exec_mask
    REQUIRE(rig.warp().get_active_count() == c);      // no set_active_mask
    REQUIRE(rig.warp().get_last_fence_position() ==
            WarpContext::kFenceBefore);  // ONLY state mutated
}
