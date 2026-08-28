// tests/integration/tcgen05/test_alloc_dealloc_relinquish.cpp
// Phase 1.x of implement-tcgen05-handlers-extended (ADR-0016, Oracle 2026-07-09):
// integration tests for the 3 alloc-family tcgen05.* handlers.
//
// Verifies processTcgen05Alloc/Dealloc/Relinquish end-to-end on a
// real CTAContext, exercising the read-only TmemAllocator API that
// was made thread-safe in this change (Fix #1).

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

#include <stdexcept>

namespace {

static void init_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

class TestRig {
public:
    TestRig() {
        init_factory_once();
        sm_ = std::make_unique<SMContext>(/*num_warps=*/1, /*warp_size=*/32,
                                          /*max_ctas=*/1, /*shared_mem=*/4096);
        cta_ = std::make_unique<CTAContext>();
        warp_ = std::make_unique<WarpContext>();
        thread_ = std::make_unique<ThreadContext>();

        warp_->set_warp_id(0);
        warp_->set_cta_context(cta_.get());
        thread_->set_warp_context(warp_.get());
    }

    CTAContext& cta() { return *cta_; }
    WarpContext& warp() { return *warp_; }
    ThreadContext& thread() { return *thread_; }

private:
    std::unique_ptr<SMContext> sm_;
    std::unique_ptr<CTAContext> cta_;
    std::unique_ptr<WarpContext> warp_;
    std::unique_ptr<ThreadContext> thread_;
};

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

TEST_CASE("processTcgen05Alloc allocates slot via TmemAllocator",
          "[integration][tcgen05][alloc][handler]") {
    TestRig rig;
    auto& alloc = rig.cta().tmem_allocator();

    REQUIRE(alloc.active_allocation_count() == 0);
    ptxsim::processTcgen05Alloc(&rig.thread(), make_alloc_instr());
    REQUIRE(alloc.active_allocation_count() == 1);
    REQUIRE(alloc.is_allocated_start(0));
    REQUIRE(alloc.is_allocated(0));
}

TEST_CASE("processTcgen05Alloc repeated calls allocate distinct slots",
          "[integration][tcgen05][alloc][handler]") {
    TestRig rig;
    auto& alloc = rig.cta().tmem_allocator();

    ptxsim::processTcgen05Alloc(&rig.thread(), make_alloc_instr());
    ptxsim::processTcgen05Alloc(&rig.thread(), make_alloc_instr());
    ptxsim::processTcgen05Alloc(&rig.thread(), make_alloc_instr());

    REQUIRE(alloc.active_allocation_count() == 3);
    REQUIRE(alloc.is_allocated_start(0));
    REQUIRE(alloc.is_allocated_start(1));
    REQUIRE(alloc.is_allocated_start(2));
}

TEST_CASE("processTcgen05Alloc throws on cta_group::2",
          "[integration][tcgen05][alloc][handler][cta_group_2]") {
    TestRig rig;
    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Alloc(&rig.thread(), make_alloc_instr(2)),
        UnsupportedInstructionException);
}

TEST_CASE("processTcgen05Alloc throws when permit relinquished",
          "[integration][tcgen05][alloc][handler][permit]") {
    TestRig rig;
    rig.warp().set_allocate_permit(false);
    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Alloc(&rig.thread(), make_alloc_instr()),
        std::runtime_error);
}

TEST_CASE("processTcgen05Dealloc releases active allocation",
          "[integration][tcgen05][dealloc][handler]") {
    TestRig rig;
    auto& alloc = rig.cta().tmem_allocator();

    ptxsim::processTcgen05Alloc(&rig.thread(), make_alloc_instr());
    REQUIRE(alloc.active_allocation_count() == 1);

    ptxsim::processTcgen05Dealloc(&rig.thread(), make_dealloc_instr());
    REQUIRE(alloc.active_allocation_count() == 0);
    REQUIRE_FALSE(alloc.is_allocated(0));
}

TEST_CASE("processTcgen05Dealloc round-trip with two allocations",
          "[integration][tcgen05][dealloc][handler]") {
    TestRig rig;
    auto& alloc = rig.cta().tmem_allocator();

    ptxsim::processTcgen05Alloc(&rig.thread(), make_alloc_instr());
    ptxsim::processTcgen05Alloc(&rig.thread(), make_alloc_instr());
    REQUIRE(alloc.active_allocation_count() == 2);

    ptxsim::processTcgen05Dealloc(&rig.thread(), make_dealloc_instr());
    REQUIRE(alloc.active_allocation_count() == 1);
    REQUIRE_FALSE(alloc.is_allocated_start(0));
    REQUIRE(alloc.is_allocated_start(1));
}

TEST_CASE("processTcgen05Dealloc throws when no allocations active",
          "[integration][tcgen05][dealloc][handler]") {
    TestRig rig;
    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Dealloc(&rig.thread(), make_dealloc_instr()),
        std::runtime_error);
}

TEST_CASE("processTcgen05Dealloc throws on cta_group::2",
          "[integration][tcgen05][dealloc][handler][cta_group_2]") {
    TestRig rig;
    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Dealloc(&rig.thread(), make_dealloc_instr(2)),
        UnsupportedInstructionException);
}

TEST_CASE("processTcgen05Relinquish sets permit to false",
          "[integration][tcgen05][relinquish][handler]") {
    TestRig rig;
    REQUIRE(rig.warp().get_allocate_permit() == true);

    ptxsim::processTcgen05Relinquish(&rig.thread(), make_relinquish_instr());
    REQUIRE(rig.warp().get_allocate_permit() == false);
}

TEST_CASE("processTcgen05Relinquish is idempotent",
          "[integration][tcgen05][relinquish][handler]") {
    TestRig rig;
    ptxsim::processTcgen05Relinquish(&rig.thread(), make_relinquish_instr());
    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Relinquish(&rig.thread(), make_relinquish_instr()));
    REQUIRE(rig.warp().get_allocate_permit() == false);
}

TEST_CASE("processTcgen05Relinquish throws on cta_group::2",
          "[integration][tcgen05][relinquish][handler][cta_group_2]") {
    TestRig rig;
    auto instr = make_relinquish_instr(2);
    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Relinquish(&rig.thread(), instr),
        UnsupportedInstructionException);
}

TEST_CASE("Full alloc→use→dealloc→relinquish workflow",
          "[integration][tcgen05][workflow][handler]") {
    TestRig rig;
    auto& alloc = rig.cta().tmem_allocator();

    ptxsim::processTcgen05Alloc(&rig.thread(), make_alloc_instr());
    ptxsim::processTcgen05Alloc(&rig.thread(), make_alloc_instr());
    REQUIRE(alloc.active_allocation_count() == 2);

    ptxsim::processTcgen05Dealloc(&rig.thread(), make_dealloc_instr());
    REQUIRE(alloc.active_allocation_count() == 1);

    ptxsim::processTcgen05Relinquish(&rig.thread(), make_relinquish_instr());
    REQUIRE(rig.warp().get_allocate_permit() == false);

    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Alloc(&rig.thread(), make_alloc_instr()),
        std::runtime_error);
}