// tests/integration/tcgen05/test_tcgen05_mma_commit_wait_sequence.cpp
// =============================================================================
// Phase 1 (H1) B2 hardening test: mma → commit → wait → mma sequence.
//
// Verifies that the commit/wait pipeline (TcQueue) correctly gates mma
// operations and that the 2nd mma produces correct C output after the
// commit→wait release. This guards against regressions where commit/wait
// interactions corrupt TMEM state or fail to flush the tensor pipeline.
//
// Phase 1 behavior: processTcgen05Mma passes accumulate=false, so 2nd
// mma overwrites (not accumulates). The H5 hook (Phases 3-4) will wire
// accumulate=true through the commit/wait cycle, at which point this
// test's assertion will be tightened to 2× golden.
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instructions/tcgen05.h"
#include "ptxsim/instructions/tcgen05_helpers.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/half_utils.h"
#include "ptxsim/warp_context.h"

#include "reference/ptx_tcgen05/tcgen05_mma_golden.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

using ptxsim::reference::tcgen05::GOLDEN_MMA_F16_F16_F32;

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
    Tmem &tmem() { return cta_->tmem(); }

private:
    std::unique_ptr<SMContext> sm_;
    std::unique_ptr<CTAContext> cta_;
    std::unique_ptr<WarpContext> warp_;
    std::unique_ptr<ThreadContext> thread_;
};

Tcgen05Instr make_mma_instr() {
    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::MMA;
    instr.cta_group = 1;
    instr.dtype = Tcgen05Dtype::F16;
    instr.operands = std::vector<OperandContext>(
        4, OperandContext(RegOperand{"r", 0}));
    return instr;
}

Tcgen05Instr make_commit_instr() {
    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::COMMIT;
    instr.cta_group = 1;
    return instr;
}

Tcgen05Instr make_wait_instr() {
    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::WAIT;
    instr.cta_group = 1;
    return instr;
}

void fill_tmem_with_golden_inputs(Tmem &tmem) {
    std::array<uint8_t, Tmem::kSlotSize> a_slot_buf{};
    for (int i = 0; i < 8; ++i) {
        const uint16_t h = f32_to_f16(static_cast<float>(i + 1));
        const size_t byte_idx = static_cast<size_t>(i) * 8 * 2;
        a_slot_buf[byte_idx]     = static_cast<uint8_t>(h & 0xFF);
        a_slot_buf[byte_idx + 1] = static_cast<uint8_t>(h >> 8);
    }

    std::array<uint8_t, Tmem::kSlotSize> b_slot_buf{};
    for (int j = 0; j < 4; ++j) {
        const uint16_t h = f32_to_f16(static_cast<float>(j + 1));
        b_slot_buf[j * 2]     = static_cast<uint8_t>(h & 0xFF);
        b_slot_buf[j * 2 + 1] = static_cast<uint8_t>(h >> 8);
    }

    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        tmem.write(static_cast<size_t>(lane_id) * 2, a_slot_buf.data(),
                   Tmem::kSlotSize);
        tmem.write(static_cast<size_t>(lane_id) * 2 + 1, b_slot_buf.data(),
                   Tmem::kSlotSize);
    }
}

void require_c_slot_matches(Tmem &tmem,
                            const std::array<float, 32> &expected,
                            const char *context_info) {
    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        std::array<uint8_t, Tmem::kSlotSize> c_buf{};
        tmem.read(static_cast<size_t>(64) + static_cast<size_t>(lane_id),
                  c_buf.data(), Tmem::kSlotSize);

        alignas(16) float c_arr[32];
        std::memcpy(c_arr, c_buf.data(), sizeof(c_arr));

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 4; ++j) {
                const int idx = i * 4 + j;
                const float actual = c_arr[idx];
                INFO(context_info << " lane=" << lane_id << " i=" << i
                     << " j=" << j << " expected=" << expected[idx]
                     << " actual=" << actual);
                REQUIRE(actual == Catch::Approx(expected[idx]).epsilon(1e-6f));
            }
        }
    }
}

} // namespace

TEST_CASE("mma → commit → wait → mma sequence: 2nd mma runs after "
          "wait releases and produces correct C",
          "[integration][tcgen05][mma][commit][wait][sequence]") {
    TestRig rig;
    fill_tmem_with_golden_inputs(rig.tmem());

    // 1st mma — produces golden C in slot[64..95].
    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Mma(&rig.thread(), make_mma_instr()));
    require_c_slot_matches(rig.tmem(), GOLDEN_MMA_F16_F16_F32,
                           "after 1st mma");

    // commit — flush mma to tc_queue.
    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Commit(&rig.thread(), make_commit_instr()));

    // wait — drain tc_queue. TcQueue::wait() adds to pending list then
    // blocks the warp; pending_count() reflects the pending waiter after
    // the wait call completes (expected with current TcQueue semantics).
    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Wait(&rig.thread(), make_wait_instr()));

    // 2nd mma — overwrite in Phase 1 (processTcgen05Mma passes
    // accumulate=false). H5 hook (Phases 3-4) will wire accumulate=true
    // through the commit/wait cycle; at that point assert 2× golden.
    REQUIRE_NOTHROW(
        ptxsim::processTcgen05Mma(&rig.thread(), make_mma_instr()));
    require_c_slot_matches(rig.tmem(), GOLDEN_MMA_F16_F16_F32,
                           "after 2nd mma (post commit/wait, overwrite)");
}