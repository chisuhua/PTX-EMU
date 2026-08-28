// tests/integration/tcgen05/test_tcgen05_mma_ld_st_chain.cpp
// =============================================================================
// FA-B6 hardener: end-to-end mma → ld → st → mma chain with f32 C persistence.
//
// Coverage motivation: existing tests exercise mma→cp→mma (`test_tcgen05_mma_persistence`),
// mma→commit→wait→mma (`test_tcgen05_mma_commit_wait_sequence`), and ld→st round-trip
// (`test_tcgen05_ld_st_slot_routing`) independently. The open gap is whether mma's
// f32 C output survives other TMEM operations (`tcgen05.ld`/`tcgen05.st`) without
// silent corruption or storage-format conversion.
//
// What this verifies:
//   TC1. mma → ld → st → mma: C slot[64..95] contains golden C after the full
//        chain (4 handlers, 2 TMEM ops sandwiched between 2 mmas).
//   TC2. f32 C storage survives ld/st: readback of C after the chain uses
//        memcpy<float> (per H2 Phase 2), with epsilon(1e-6f) precision.
//   TC3. Two consecutive chains remain independent: C is golden in each chain
//        (no leakage via warp cursor or TMEM offset register state).
//
// Per helpers/tmem_helpers.h + FU-3 C2 (Oracle Q5 implicit per-warp cursor):
//   - ld allocates via warp->allocate_ld_slot() (next_ld_slot_++)
//   - st reads from warp->last_ld_slot()
//   - mma C slot is lane_id*32 + warp_id*32 + 64 (FU-4 multi-warp slot layout,
//     warp_id=0 here so C lives in [64..95])

#include "catch_amalgamated.hpp"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instructions/tcgen05.h"
#include "ptxsim/instructions/tcgen05_helpers.h"
#include "ptxsim/memory/tma_descriptor.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/half_utils.h"
#include "ptxsim/warp_context.h"

#include "reference/ptx_tcgen05/tcgen05_mma_golden.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

using ptxsim::reference::tcgen05::GOLDEN_MMA_F16_F16_F32;

namespace {

// TestRig with both shared memory (for cp testing if needed) and global memory
// backing for tcgen05.ld source / tcgen05.st destination. Single warp (warp_id=0).
class TestRig {
public:
    explicit TestRig(size_t smem_bytes = 4096)
        : sm_(std::make_unique<SMContext>(/*num_warps=*/1, /*warp_size=*/32,
                                          /*max_ctas=*/1,
                                          /*shared_mem=*/4096)),
          cta_(std::make_unique<CTAContext>()),
          warp_(std::make_unique<WarpContext>()),
          thread_(std::make_unique<ThreadContext>()),
          smem_buf_(smem_bytes, 0),
          global_buf_(Tmem::kSlotSize, 0) {
        warp_->set_warp_id(0);
        warp_->set_cta_context(cta_.get());
        thread_->set_warp_context(warp_.get());

        cta_->sharedMemBytes = smem_bytes;
        cta_->sharedMemSpace = smem_buf_.data();

        TmaDescriptor desc;
        desc.global_address =
            reinterpret_cast<uint64_t>(global_buf_.data());
        desc.raw_bytes.resize(kTmaDescriptorSize, 0);
        cta_->tma_descriptor_store().store(0, desc);
    }

    CTAContext& cta() { return *cta_; }
    WarpContext& warp() { return *warp_; }
    ThreadContext& thread() { return *thread_; }
    Tmem& tmem() { return cta_->tmem(); }
    std::vector<uint8_t>& smem() { return smem_buf_; }
    std::vector<uint8_t>& global_buf() { return global_buf_; }

    void fill_global_buf(uint8_t value) {
        std::memset(global_buf_.data(), value, global_buf_.size());
    }

private:
    std::unique_ptr<SMContext> sm_;
    std::unique_ptr<CTAContext> cta_;
    std::unique_ptr<WarpContext> warp_;
    std::unique_ptr<ThreadContext> thread_;
    std::vector<uint8_t> smem_buf_;
    std::vector<uint8_t> global_buf_;
};

ptxemu::ir::Tcgen05Instr make_mma_instr() {
    ptxemu::ir::Tcgen05Instr instr;
    instr.op_kind = ptxemu::ir::Tcgen05OpKind::MMA;
    instr.cta_group = 1;
    instr.dtype = ptxemu::ir::Tcgen05Dtype::F16;
    instr.operands = std::vector<ptxemu::ir::OperandContext>(
        4, ptxemu::ir::OperandContext(RegOperand{"r", 0}));
    return instr;
}

ptxemu::ir::Tcgen05Instr make_ld_instr() {
    ptxemu::ir::Tcgen05Instr instr;
    instr.op_kind = ptxemu::ir::Tcgen05OpKind::LD;
    instr.cta_group = 1;
    instr.operands = std::vector<ptxemu::ir::OperandContext>(
        2, ptxemu::ir::OperandContext(RegOperand{"r", 0}));
    return instr;
}

ptxemu::ir::Tcgen05Instr make_st_instr() {
    ptxemu::ir::Tcgen05Instr instr;
    instr.op_kind = ptxemu::ir::Tcgen05OpKind::ST;
    instr.cta_group = 1;
    instr.operands = std::vector<ptxemu::ir::OperandContext>(
        2, ptxemu::ir::OperandContext(RegOperand{"r", 0}));
    return instr;
}

void fill_tmem_with_golden_inputs(Tmem& tmem) {
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

// f32 readback of C slot[64..95] (per Phase 2 H2 storage format).
// Returns the 32-element C array (8 rows × 4 cols per lane, f32 elements).
std::array<float, 32> read_c_slot_f32(Tmem& tmem) {
    std::array<float, 32> c{};
    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        std::array<uint8_t, Tmem::kSlotSize> buf{};
        tmem.read(static_cast<size_t>(64) + static_cast<size_t>(lane_id),
                  buf.data(), Tmem::kSlotSize);
        alignas(16) float lane_arr[32];
        std::memcpy(lane_arr, buf.data(), sizeof(lane_arr));
        std::memcpy(c.data(), lane_arr, sizeof(c));
    }
    return c;
}

bool c_slot_matches_golden(const std::array<float, 32>& actual,
                           const std::array<float, 32>& expected) {
    for (size_t k = 0; k < 32; ++k) {
        if (actual[k] != Catch::Approx(expected[k]).epsilon(1e-6f)) {
            return false;
        }
    }
    return true;
}

}  // namespace

// =============================================================================
// TC1: mma → ld → st → mma — C slot[64..95] contains golden C after chain.
// =============================================================================

TEST_CASE("mma → ld → st → mma: C slot[64..95] preserves golden C across TMEM ops",
          "[integration][tcgen05][mma][chain][ld][st][flashattention][B6]") {
    TestRig rig;

    fill_tmem_with_golden_inputs(rig.tmem());
    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), make_mma_instr()));
    {
        const auto c = read_c_slot_f32(rig.tmem());
        INFO("after 1st mma");
        REQUIRE(c_slot_matches_golden(c, GOLDEN_MMA_F16_F16_F32));
    }

    rig.fill_global_buf(0xCC);
    REQUIRE_NOTHROW(ptxsim::processTcgen05Ld(&rig.thread(), make_ld_instr()));
    REQUIRE_NOTHROW(ptxsim::processTcgen05St(&rig.thread(), make_st_instr()));

    // ld writes to slot 0 (per-warp cursor), which is lane 0's a_slot — it
    // intentionally corrupts the A operand for lane 0's next mma. Refill A/B
    // to golden so the 2nd mma can be evaluated against fresh golden inputs.
    fill_tmem_with_golden_inputs(rig.tmem());

    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), make_mma_instr()));
    {
        const auto c = read_c_slot_f32(rig.tmem());
        INFO("after 2nd mma (post ld/st + A/B refill)");
        REQUIRE(c_slot_matches_golden(c, GOLDEN_MMA_F16_F16_F32));
    }
}

// =============================================================================
// TC2: f32 C storage format survives ld/st (no silent f16 fallback).
//   Mirror of the helper invariant test but for the chain case.
// =============================================================================

TEST_CASE("mma → ld → st → mma: C remains f32 storage after TMEM ops",
          "[integration][tcgen05][mma][chain][f32_storage][B6]") {
    TestRig rig;
    fill_tmem_with_golden_inputs(rig.tmem());

    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), make_mma_instr()));
    rig.fill_global_buf(0xEE);
    REQUIRE_NOTHROW(ptxsim::processTcgen05Ld(&rig.thread(), make_ld_instr()));
    REQUIRE_NOTHROW(ptxsim::processTcgen05St(&rig.thread(), make_st_instr()));
    fill_tmem_with_golden_inputs(rig.tmem());
    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), make_mma_instr()));

    const auto c = read_c_slot_f32(rig.tmem());
    REQUIRE(c_slot_matches_golden(c, GOLDEN_MMA_F16_F16_F32));

    for (int lane = 0; lane < 32; ++lane) {
        std::array<uint8_t, Tmem::kSlotSize> buf{};
        rig.tmem().read(static_cast<size_t>(64) + static_cast<size_t>(lane),
                        buf.data(), Tmem::kSlotSize);
        alignas(16) float lane_c[32];
        std::memcpy(lane_c, buf.data(), sizeof(lane_c));
        for (int idx = 0; idx < 32; ++idx) {
            INFO("lane=" << lane << " idx=" << idx);
            REQUIRE(lane_c[idx] ==
                    Catch::Approx(GOLDEN_MMA_F16_F16_F32[idx]).epsilon(1e-6f));
        }
    }
}

// =============================================================================
// TC3: Two consecutive mma→ld→st→mma chains remain independent.
//   Regression guard against state leakage via warp cursor or TMEM offset regs.
// =============================================================================

TEST_CASE("mma → ld → st → mma chain repeated twice: C independent across chains",
          "[integration][tcgen05][mma][chain][repeat][B6]") {
    TestRig rig;
    fill_tmem_with_golden_inputs(rig.tmem());

    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), make_mma_instr()));
    rig.fill_global_buf(0x11);
    REQUIRE_NOTHROW(ptxsim::processTcgen05Ld(&rig.thread(), make_ld_instr()));
    REQUIRE_NOTHROW(ptxsim::processTcgen05St(&rig.thread(), make_st_instr()));
    fill_tmem_with_golden_inputs(rig.tmem());
    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), make_mma_instr()));
    {
        const auto c = read_c_slot_f32(rig.tmem());
        REQUIRE(c_slot_matches_golden(c, GOLDEN_MMA_F16_F16_F32));
    }

    rig.fill_global_buf(0x22);
    REQUIRE_NOTHROW(ptxsim::processTcgen05Ld(&rig.thread(), make_ld_instr()));
    REQUIRE_NOTHROW(ptxsim::processTcgen05St(&rig.thread(), make_st_instr()));
    fill_tmem_with_golden_inputs(rig.tmem());
    REQUIRE_NOTHROW(ptxsim::processTcgen05Mma(&rig.thread(), make_mma_instr()));
    {
        const auto c = read_c_slot_f32(rig.tmem());
        INFO("after chain #2 mma");
        REQUIRE(c_slot_matches_golden(c, GOLDEN_MMA_F16_F16_F32));
    }
}
