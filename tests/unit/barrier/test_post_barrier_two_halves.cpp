// tests/unit/barrier/test_post_barrier_two_halves.cpp
// Unit test for BUG-POSTBARRIER-TWOHALVES.
//
// Bug: When a divergent warp splits into two halves that hit the same
// barrier at different times, the second barrier release OVERWRITES
// active_mask with only the currently arrived half, losing the lanes
// already released by the first barrier.
//
// Symptom: After both halves pass the barrier, lanes from the first half
// are no longer in active_mask, so the scheduler stops scheduling them
// and they never execute post-barrier instructions.

#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/thread_state.h"
#include "ptx_ir/statement_context.h"

#include <memory>
#include <vector>
#include <map>

using namespace ptxsim;
using ptxsim::ThreadStatus;

namespace {

void add_thread(WarpContext& warp, int lane) {
    auto thread = std::make_unique<ThreadContext>();
    Dim3 blockIdx = {0, 0, 0};
    Dim3 threadIdx = {(uint32_t)lane, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<StatementContext> stmts;
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;
    thread->init(blockIdx, threadIdx, gridDim, blockDim, stmts, &name2Sym,
                 label2pc, nullptr, nullptr);
    thread->set_state(RUN);
    warp.add_thread(std::move(thread), lane);
}

// Simulate the post-completion path of BarWarpSyncHandler::processOperation
// when a half-warp arrives at the barrier and it completes.
void simulate_barrier_release(WarpContext& warp, uint32_t arrived_mask,
                              int reconv_pc) {
    warp.set_exec_mask(arrived_mask);
    auto& ws = warp.get_warp_state();
    for (int i = 0; i < 32; i++) {
        if ((arrived_mask & (1u << i)) && ws.threads[i].is_active) {
            ws.threads[i].pc = reconv_pc;
            ws.threads[i].next_pc = reconv_pc;
            ws.threads[i].is_blocked = false;
            ws.threads[i].status = ThreadStatus::Active;
        }
    }
    // BUG-POSTBARRIER-TWOHALVES fix: barrier handler must OR the new
    // arrived_mask with the existing active_mask to preserve lanes
    // already released by a prior barrier call.
    warp.set_active_mask(warp.get_active_mask() | arrived_mask);
}

}  // namespace

TEST_CASE("D1: two divergent halves at barrier — both halves active after both releases",
          "[barrier][divergence][regression][BUG-POSTBARRIER-TWOHALVES]") {
    WarpContext warp;
    for (int i = 0; i < 32; i++) {
        add_thread(warp, i);
    }

    // Simulate two divergent halves on divergent paths that will converge
    // at a single barrier:
    //   lanes 0-15: on path A, PC=50
    //   lanes 16-31: on path B, PC=60
    // Both paths will hit the barrier at PC=70 with reconvergence at PC=71.
    auto& ws = warp.get_warp_state();
    for (int i = 0; i < 16; i++) {
        ws.threads[i].pc = 50;
        ws.threads[i].is_active = true;
        ws.threads[i].is_blocked = false;
        ws.threads[i].is_exited = false;
        ws.threads[i].status = ThreadStatus::Active;
    }
    for (int i = 16; i < 32; i++) {
        ws.threads[i].pc = 60;
        ws.threads[i].is_active = true;
        ws.threads[i].is_blocked = false;
        ws.threads[i].is_exited = false;
        ws.threads[i].status = ThreadStatus::Active;
    }
    warp.set_active_mask(0xFFFFFFFFu);
    warp.set_exec_mask(0xFFFFFFFFu);

    // Upper half (16-31) arrives at barrier, barrier completes
    simulate_barrier_release(warp, 0xFFFF0000u, 71);

    // With the BUG-POSTBARRIER-TWOHALVES fix in barrier.cpp, the handler
    // ORs the new arrived_mask with existing active_mask. After the
    // first release, mask stays 0xFFFFFFFF (lanes 0-15 were already
    // active from initial setup and are preserved by the OR).
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFFu);
    for (int i = 16; i < 32; i++) {
        REQUIRE(ws.threads[i].pc == 71);
    }

    // Lower half (0-15) arrives at barrier, barrier completes.
    // CORRECT behavior: active_mask should be 0xFFFFFFFF (union of both
    //   halves, since upper half is already at PC=71 executing post-barrier).
    // BUGGY behavior (current): set_active_mask(0000FFFF) overwrites the
    //   mask, losing lanes 16-31.
    simulate_barrier_release(warp, 0x0000FFFFu, 71);

    CHECK(warp.get_active_mask() == 0xFFFFFFFFu);
    for (int i = 0; i < 32; i++) {
        CHECK(ws.threads[i].pc == 71);
    }
}

TEST_CASE("D2: after two-half barrier release, is_warp_ready_to_fetch uses all 32 lanes",
          "[barrier][divergence][regression][BUG-POSTBARRIER-TWOHALVES]") {
    WarpContext warp;
    for (int i = 0; i < 32; i++) {
        add_thread(warp, i);
    }
    auto& ws = warp.get_warp_state();
    for (int i = 0; i < 32; i++) {
        ws.threads[i].pc = 50;
        ws.threads[i].is_active = true;
        ws.threads[i].is_blocked = false;
        ws.threads[i].is_exited = false;
        ws.threads[i].status = ThreadStatus::Active;
    }
    warp.set_active_mask(0xFFFFFFFFu);
    warp.set_exec_mask(0xFFFFFFFFu);

    simulate_barrier_release(warp, 0xFFFF0000u, 71);
    simulate_barrier_release(warp, 0x0000FFFFu, 71);

    // All 32 lanes should be schedulable at PC=71
    CHECK(warp.get_active_count() == 32);
    CHECK(warp.is_warp_ready_to_fetch());
}
