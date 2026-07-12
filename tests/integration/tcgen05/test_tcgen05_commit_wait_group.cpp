// tests/integration/tcgen05/test_tcgen05_commit_wait_group.cpp
// =============================================================================
// FU-1 Oracle C3 fix: commit/wait handler cta_group routing.
//
// Verifies that processTcgen05Commit + processTcgen05Wait use instr.cta_group
// from IR (not hardcoded group_id=1). Pre-fix, both handlers hardcoded
// group_id=1 and explicitly cast `(void)instr;`.
//
// Tests:
//   T1. commit(2) advances counter to 2 → wait(2) returns immediately
//   T2. wait(2) without prior commit blocks (deadlock detection via async)
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

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <vector>

namespace {

class TestRig {
public:
    TestRig()
        : sm_(std::make_unique<SMContext>(1, 32, 1, 4096)),
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
    TcQueue &tc_queue() { return cta_->tc_queue(); }

private:
    std::unique_ptr<SMContext> sm_;
    std::unique_ptr<CTAContext> cta_;
    std::unique_ptr<WarpContext> warp_;
    std::unique_ptr<ThreadContext> thread_;
};

Tcgen05Instr make_commit_instr(uint32_t cta_group) {
    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::COMMIT;
    instr.cta_group = cta_group;
    return instr;
}

Tcgen05Instr make_wait_instr(uint32_t cta_group) {
    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::WAIT;
    instr.cta_group = cta_group;
    return instr;
}

} // namespace

TEST_CASE("processTcgen05Commit routes instr.cta_group to TcQueue (FU-1 C3)",
          "[integration][tcgen05][commit][cta_group][FU-1]") {
    TestRig rig;

    // commit(2) → handler should call tc_queue.commit(2), not commit(1)
    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Commit(&rig.thread(), make_commit_instr(2)));

    // Counter should be 2 (not 1)
    REQUIRE(rig.tc_queue().current_counter() == 2u);
}

TEST_CASE("processTcgen05Commit + Wait with cta_group=2 succeeds (FU-1 C3)",
          "[integration][tcgen05][commit][wait][cta_group][FU-1]") {
    TestRig rig;

    // Setup: warp lane 0 at some PC
    rig.warp().advance_thread_pc(0, 100);

    // commit(2) → counter=2
    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Commit(&rig.thread(), make_commit_instr(2)));

    // wait(2) → counter(2) ≥ waited(2) → returns immediately
    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Wait(&rig.thread(), make_wait_instr(2)));

    // lane 0 should NOT be blocked (counter already satisfied)
    auto& ts = rig.warp().get_warp_state().threads[0];
    REQUIRE(ts.is_blocked == false);
}

TEST_CASE("processTcgen05Wait with cta_group=2 blocks when counter < 2 (FU-1 C3)",
          "[integration][tcgen05][commit][wait][cta_group][FU-1]") {
    TestRig rig;

    // Setup: warp lane 0 at PC 200
    rig.warp().advance_thread_pc(0, 200);

    // wait(2) without prior commit → should block
    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Wait(&rig.thread(), make_wait_instr(2)));

    auto& ts = rig.warp().get_warp_state().threads[0];
    REQUIRE(ts.is_blocked == true);

    // Verify pending waiter exists
    REQUIRE(rig.tc_queue().pending_count() == 1);

    // Release by commit(2)
    rig.tc_queue().commit(2);

    REQUIRE(ts.is_blocked == false);
    // PC should advance to 201 (captured at wait time as 200+1)
    REQUIRE(rig.warp().get_thread_pc(0) == 201);
}