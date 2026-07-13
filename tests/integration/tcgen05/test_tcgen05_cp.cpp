// tests/integration/tcgen05/test_tcgen05_cp.cpp
// =============================================================================
// Phase 2 of tcgen05-cp-test-coverage-and-exception-cleanup:
// integration tests for tcgen05.cp SMEM → TMEM copy.
//
// Verifies processTcgen05Cp end-to-end on a real CTAContext, exercising
// the 128-byte copy path + out-of-bounds exception path. Unit-level
// coverage of helpers lives in tests/unit/tcgen05/test_tcgen05_cp.cpp.
//
// Spec reference:
// openspec/changes/tcgen05-cp-test-coverage-and-exception-cleanup/
//                 specs/tcgen05-cp-test-coverage/spec.md
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instructions/tcgen05.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

class TestRig {
public:
    explicit TestRig(size_t smem_bytes = 4096)
        : sm_(std::make_unique<SMContext>(/*num_warps=*/1, /*warp_size=*/32,
                                          /*max_ctas=*/1, /*shared_mem=*/4096)),
          cta_(std::make_unique<CTAContext>()),
          warp_(std::make_unique<WarpContext>()),
          thread_(std::make_unique<ThreadContext>()), smem_buf_(smem_bytes, 0) {
        warp_->set_warp_id(0);
        warp_->set_cta_context(cta_.get());
        thread_->set_warp_context(warp_.get());

        cta_->sharedMemBytes = smem_bytes;
        cta_->sharedMemSpace = smem_buf_.data();
    }

    CTAContext &cta() { return *cta_; }
    WarpContext &warp() { return *warp_; }
    ThreadContext &thread() { return *thread_; }
    std::vector<uint8_t> &smem() { return smem_buf_; }
    Tmem &tmem() { return cta_->tmem(); }

private:
    std::unique_ptr<SMContext> sm_;
    std::unique_ptr<CTAContext> cta_;
    std::unique_ptr<WarpContext> warp_;
    std::unique_ptr<ThreadContext> thread_;
    std::vector<uint8_t> smem_buf_;
};

Tcgen05Instr make_cp_instr_with_smem_offset(uint32_t smem_offset) {
    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::CP;
    instr.cta_group = 1;

    AddrOperand dst_dummy;
    dst_dummy.space = AddrOperand::Space::SHARED;
    dst_dummy.offsetType = AddrOperand::OffsetType::IMMEDIATE;
    dst_dummy.immediateOffset = "0";
    instr.operands.push_back(OperandContext(dst_dummy));

    AddrOperand src;
    src.space = AddrOperand::Space::SHARED;
    src.offsetType = AddrOperand::OffsetType::IMMEDIATE;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%u", smem_offset);
    src.immediateOffset = buf;
    instr.operands.push_back(OperandContext(src));

    return instr;
}

} // namespace

// ============================================================================
// Happy path: 128-byte copy from SMEM to TMEM slot 0
// ============================================================================

TEST_CASE("processTcgen05Cp copies 128 bytes from SMEM offset 0 to TMEM cp slot",
          "[integration][tcgen05][cp][handler][happy_path]") {
    TestRig rig(4096);

    // Seed SMEM with a recognizable byte pattern.
    for (size_t i = 0; i < Tmem::kSlotSize; ++i) {
        rig.smem()[i] = static_cast<uint8_t>(i ^ 0x5A);
    }
    uint8_t expected[Tmem::kSlotSize];
    std::memcpy(expected, rig.smem().data(), Tmem::kSlotSize);

    REQUIRE_NOTHROW(ptxsim::processTcgen05Cp(
        &rig.thread(), make_cp_instr_with_smem_offset(0)));

    // FU-3 C2: cp writes to slot 32 (base 32 + cursor 0) instead of
    // hardcoded slot 0. Verified by reading from the cp target slot.
    uint8_t actual[Tmem::kSlotSize] = {0};
    rig.tmem().read(32, actual, Tmem::kSlotSize);
    REQUIRE(std::memcmp(actual, expected, Tmem::kSlotSize) == 0);
}

TEST_CASE("processTcgen05Cp copies 128 bytes from non-zero SMEM offset to TMEM "
          "cp slot",
          "[integration][tcgen05][cp][handler][happy_path][offset]") {
    constexpr size_t kOffset = 256;
    TestRig rig(4096);

    for (size_t i = 0; i < Tmem::kSlotSize; ++i) {
        rig.smem()[kOffset + i] = static_cast<uint8_t>(0xC0 + (i & 0x0F));
    }
    uint8_t expected[Tmem::kSlotSize];
    std::memcpy(expected, rig.smem().data() + kOffset, Tmem::kSlotSize);

    REQUIRE_NOTHROW(ptxsim::processTcgen05Cp(
        &rig.thread(), make_cp_instr_with_smem_offset(kOffset)));

    uint8_t actual[Tmem::kSlotSize] = {0};
    rig.tmem().read(32, actual, Tmem::kSlotSize);
    REQUIRE(std::memcmp(actual, expected, Tmem::kSlotSize) == 0);
}

// ============================================================================
// Out-of-bounds: exception is observed, TMEM is not corrupted
// ============================================================================

TEST_CASE("processTcgen05Cp throws when smem offset + kSlotSize exceeds "
          "sharedMemBytes",
          "[integration][tcgen05][cp][handler][oob]") {
    constexpr size_t kSmemBytes = 256;
    TestRig rig(kSmemBytes);

    // 256 + 128 = 384 > 256 → out of bounds.
    REQUIRE_THROWS_AS(ptxsim::processTcgen05Cp(
                          &rig.thread(), make_cp_instr_with_smem_offset(256)),
                      std::runtime_error);

    // Verify TMEM was not corrupted: slot 0 should remain its initial state
    // (all zeros per the default Tmem ctor).
    uint8_t actual[Tmem::kSlotSize] = {0xFF};
    rig.tmem().read(0, actual, Tmem::kSlotSize);
    for (size_t i = 0; i < Tmem::kSlotSize; ++i) {
        REQUIRE(actual[i] == 0);
    }
}