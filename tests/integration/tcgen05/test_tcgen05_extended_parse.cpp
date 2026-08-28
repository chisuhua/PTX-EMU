// tests/integration/tcgen05/test_tcgen05_extended_parse.cpp
// =============================================================================
// Phase 4 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q5-C / Q6-B):
// integration tests for the tcgen05.fence no-op marker on a real CTAContext
// + WarpContext rig (type-2 integration per AGENTS.md testing taxonomy).
//
// Verifies end-to-end that:
//   - processTcgen05Fence records position via WARP-LEVEL state mutation only
//     (no CTAContext / TmemAllocator / Smem side effects)
//   - multi-warp independent (per-WarpState fields)
//   - allocator/cp/fence interleave preserves state-modification audit invariants
//   - cta_group::2 throws through the integration dispatch path
//
// Unit-level coverage of helpers and error paths lives in
// tests/unit/ptx_ir/test_tcgen05_extended_opkind.cpp.
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/instructions/tcgen05.h"
#include "ptxsim/memory/tmem_allocator.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"

#include <memory>
#include <stdexcept>
#include <vector>

namespace {

// Mirrors the TestRig pattern from test_alloc_dealloc_relinquish.cpp. We
// instantiate a single-CTA / single-warp world so the integration test is
// repeatable across fence operations.
static void init_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

class FenceIntegrationRig {
public:
    FenceIntegrationRig() {
        init_factory_once();
        sm_ = std::make_unique<SMContext>(/*num_warps=*/2, /*warp_size=*/32,
                                          /*max_ctas=*/1, /*shared_mem=*/4096);
        cta_ = std::make_unique<CTAContext>();
        warp0_ = std::make_unique<WarpContext>();
        warp1_ = std::make_unique<WarpContext>();
        thread0_ = std::make_unique<ThreadContext>();
        thread1_ = std::make_unique<ThreadContext>();

        warp0_->set_warp_id(0);
        warp0_->set_cta_context(cta_.get());
        warp1_->set_warp_id(1);
        warp1_->set_cta_context(cta_.get());
        thread0_->set_warp_context(warp0_.get());
        thread1_->set_warp_context(warp1_.get());
    }

    CTAContext &cta() { return *cta_; }
    WarpContext &warp0() { return *warp0_; }
    WarpContext &warp1() { return *warp1_; }
    ThreadContext &thread0() { return *thread0_; }
    ThreadContext &thread1() { return *thread1_; }

private:
    std::unique_ptr<SMContext> sm_;
    std::unique_ptr<CTAContext> cta_;
    std::unique_ptr<WarpContext> warp0_;
    std::unique_ptr<WarpContext> warp1_;
    std::unique_ptr<ThreadContext> thread0_;
    std::unique_ptr<ThreadContext> thread1_;
};

ptxemu::ir::Tcgen05Instr make_fence_instr(uint32_t cta_group,
                              std::vector<ptxemu::ir::Qualifier> qualifiers) {
    ptxemu::ir::Tcgen05Instr instr;
    instr.op_kind = ptxemu::ir::Tcgen05OpKind::FENCE;
    instr.cta_group = cta_group;
    instr.qualifiers = std::move(qualifiers);
    return instr;
}

ptxemu::ir::Tcgen05Instr make_alloc_instr(uint32_t cta_group = 1) {
    ptxemu::ir::Tcgen05Instr instr;
    instr.op_kind = ptxemu::ir::Tcgen05OpKind::ALLOC;
    instr.cta_group = cta_group;
    return instr;
}

ptxemu::ir::Tcgen05Instr make_dealloc_instr(uint32_t cta_group = 1) {
    ptxemu::ir::Tcgen05Instr instr;
    instr.op_kind = ptxemu::ir::Tcgen05OpKind::DEALLOC;
    instr.cta_group = cta_group;
    return instr;
}

ptxemu::ir::Tcgen05Instr make_relinquish_instr(uint32_t cta_group = 1) {
    ptxemu::ir::Tcgen05Instr instr;
    instr.op_kind = ptxemu::ir::Tcgen05OpKind::RELINQUISH;
    instr.cta_group = cta_group;
    return instr;
}

}  // namespace

// -----------------------------------------------------------------------------
// (1) Forward path: ::before_thread_sync records position; no other state moves
// -----------------------------------------------------------------------------
TEST_CASE("processTcgen05Fence: integration ::before_thread_sync records position",
          "[integration][tcgen05][fence][handler][happy_path]") {
    FenceIntegrationRig rig;

    const auto before_permit = rig.warp0().get_allocate_permit();
    const auto before_exec = rig.warp0().get_exec_mask();
    const auto before_active = rig.warp0().get_active_count();
    const auto before_allocs = rig.cta().tmem_allocator().active_allocation_count();

    REQUIRE_NOTHROW(ptxsim::processTcgen05Fence(
        &rig.thread0(),
        make_fence_instr(1, {ptxemu::ir::Qualifier::Q_BEFORE_THREAD_SYNC})));

    REQUIRE(rig.warp0().get_last_fence_position() ==
            WarpContext::kFenceBefore);
    // Q5-C invariants: allocate_permit / exec_mask / active_count unchanged
    REQUIRE(rig.warp0().get_allocate_permit() == before_permit);
    REQUIRE(rig.warp0().get_exec_mask() == before_exec);
    REQUIRE(rig.warp0().get_active_count() == before_active);
    // No TMEM allocation triggered by fence
    REQUIRE(rig.cta().tmem_allocator().active_allocation_count() ==
            before_allocs);
}

// -----------------------------------------------------------------------------
// (2) Alloc + fence + dealloc interleave: state-modification audit passes
// -----------------------------------------------------------------------------
TEST_CASE("processTcgen05Fence: integration alloc/fence/dealloc interleave",
          "[integration][tcgen05][fence][handler][interleave]") {
    FenceIntegrationRig rig;
    auto &alloc = rig.cta().tmem_allocator();

    REQUIRE(alloc.active_allocation_count() == 0);

    ptxsim::processTcgen05Alloc(&rig.thread0(), make_alloc_instr());
    REQUIRE(alloc.active_allocation_count() == 1);

    ptxsim::processTcgen05Fence(
        &rig.thread0(),
        make_fence_instr(1, {ptxemu::ir::Qualifier::Q_AFTER_THREAD_SYNC}));
    REQUIRE(rig.warp0().get_last_fence_position() ==
            WarpContext::kFenceAfter);
    REQUIRE(alloc.active_allocation_count() == 1);  // fence did not allocate

    ptxsim::processTcgen05Dealloc(&rig.thread0(), make_dealloc_instr());
    REQUIRE(alloc.active_allocation_count() == 0);

    ptxsim::processTcgen05Relinquish(&rig.thread0(), make_relinquish_instr());
    REQUIRE(rig.warp0().get_allocate_permit() == false);

    // fence was interleaveable — position survives alloc/dealloc/relinquish
    REQUIRE(rig.warp0().get_last_fence_position() ==
            WarpContext::kFenceAfter);
}

// -----------------------------------------------------------------------------
// (3) Multi-warp independence: each WarpState has its own fence_position
// -----------------------------------------------------------------------------
TEST_CASE("processTcgen05Fence: integration multi-warp independence",
          "[integration][tcgen05][fence][handler][multi_warp]") {
    FenceIntegrationRig rig;

    ptxsim::processTcgen05Fence(
        &rig.thread0(),
        make_fence_instr(1, {ptxemu::ir::Qualifier::Q_BEFORE_THREAD_SYNC}));
    ptxsim::processTcgen05Fence(
        &rig.thread1(),
        make_fence_instr(1, {ptxemu::ir::Qualifier::Q_AFTER_THREAD_SYNC}));

    REQUIRE(rig.warp0().get_last_fence_position() ==
            WarpContext::kFenceBefore);
    REQUIRE(rig.warp1().get_last_fence_position() ==
            WarpContext::kFenceAfter);

    // Records persist independently.
    ptxsim::processTcgen05Fence(
        &rig.thread0(),
        make_fence_instr(1, {ptxemu::ir::Qualifier::Q_AFTER_THREAD_SYNC}));
    REQUIRE(rig.warp0().get_last_fence_position() ==
            WarpContext::kFenceAfter);
    REQUIRE(rig.warp1().get_last_fence_position() ==
            WarpContext::kFenceAfter);  // warp1 unchanged
}

// -----------------------------------------------------------------------------
// (4) Error path: cta_group::2 throws via integration dispatch path
// -----------------------------------------------------------------------------
TEST_CASE("processTcgen05Fence: integration cta_group::2 throws ADR-0018",
          "[integration][tcgen05][fence][handler][error][cta_group_2]") {
    FenceIntegrationRig rig;

    REQUIRE(rig.warp0().get_last_fence_position() == WarpContext::kFenceNone);
    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Fence(
            &rig.thread0(),
            make_fence_instr(2, {ptxemu::ir::Qualifier::Q_BEFORE_THREAD_SYNC})),
        UnsupportedInstructionException);
    // State must NOT be mutated before the throw
    REQUIRE(rig.warp0().get_last_fence_position() == WarpContext::kFenceNone);
}
