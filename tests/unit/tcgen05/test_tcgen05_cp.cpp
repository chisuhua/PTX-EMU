// tests/unit/tcgen05/test_tcgen05_cp.cpp
// =============================================================================
// Phase 1 of tcgen05-cp-test-coverage-and-exception-cleanup:
// unit tests for tcgen05.cp helper functions and exception paths.
//
// RED phase note: `extract_smem_offset_placeholder` and `throw_cta_group_2`
// are currently in anonymous namespace inside tcgen05_cp.cpp. The
// forward declarations below intentionally fail to link — confirming that
// these helpers are not yet reachable from tests. Phase 1.3 promotes them
// to `ptxsim` namespace so this file links successfully.
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
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"

#include <memory>
#include <string>

// -----------------------------------------------------------------------------
// Forward declarations for helpers currently in anonymous namespace.
// Phase 1.3 promotes them to `ptxsim` namespace; until then this file
// fails to link (the RED state).
// -----------------------------------------------------------------------------
namespace ptxsim {
uint32_t extract_smem_offset_placeholder(const Tcgen05Instr &instr);
[[noreturn]] void throw_cta_group_2(const char *instr_name);
} // namespace ptxsim

namespace {

// Minimal TestRig: SM + CTA + Warp + Thread. Mirrors the pattern used by
// tests/integration/tcgen05/test_alloc_dealloc_relinquish.cpp.
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

Tcgen05Instr make_cp_instr(uint32_t cta_group = 1) {
    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::CP;
    instr.cta_group = cta_group;
    return instr;
}

// Build an AddrOperand with a textual immediate offset and given space.
OperandContext make_addr_op(AddrOperand::Space space,
                            AddrOperand::OffsetType offset_type,
                            const std::string &immediate) {
    AddrOperand addr;
    addr.space = space;
    addr.offsetType = offset_type;
    addr.immediateOffset = immediate;
    return OperandContext(addr);
}

// Build a Tcgen05Instr with a dummy operands[0] and the supplied SMEM
// address at operands[1] (PTX operand order: dst, src).
Tcgen05Instr make_cp_instr_with_smem(
    AddrOperand::Space space = AddrOperand::Space::SHARED,
    AddrOperand::OffsetType offset_type = AddrOperand::OffsetType::IMMEDIATE,
    const std::string &immediate = "0") {
    Tcgen05Instr instr = make_cp_instr();
    AddrOperand dst_dummy;
    dst_dummy.space = AddrOperand::Space::SHARED;
    dst_dummy.offsetType = AddrOperand::OffsetType::IMMEDIATE;
    dst_dummy.immediateOffset = "0";
    instr.operands.push_back(OperandContext(dst_dummy));
    instr.operands.push_back(make_addr_op(space, offset_type, immediate));
    return instr;
}

} // namespace

// ============================================================================
// extract_smem_offset_placeholder helper tests
// ============================================================================

TEST_CASE("extract_smem_offset_placeholder parses hexadecimal immediate offset",
          "[unit][tcgen05][cp][helper]") {
    Tcgen05Instr instr = make_cp_instr_with_smem(
        AddrOperand::Space::SHARED, AddrOperand::OffsetType::IMMEDIATE, "0x10");
    REQUIRE(ptxsim::extract_smem_offset_placeholder(instr) == 16u);
}

TEST_CASE("extract_smem_offset_placeholder parses decimal immediate offset",
          "[unit][tcgen05][cp][helper]") {
    Tcgen05Instr instr = make_cp_instr_with_smem(
        AddrOperand::Space::SHARED, AddrOperand::OffsetType::IMMEDIATE, "32");
    REQUIRE(ptxsim::extract_smem_offset_placeholder(instr) == 32u);
}

TEST_CASE("extract_smem_offset_placeholder returns 0 for non-shared space",
          "[unit][tcgen05][cp][helper]") {
    Tcgen05Instr instr = make_cp_instr_with_smem(
        AddrOperand::Space::GLOBAL, AddrOperand::OffsetType::IMMEDIATE, "64");
    REQUIRE(ptxsim::extract_smem_offset_placeholder(instr) == 0u);
}

TEST_CASE("extract_smem_offset_placeholder returns 0 for register offset "
          "(placeholder)",
          "[unit][tcgen05][cp][helper]") {
    Tcgen05Instr instr = make_cp_instr_with_smem(
        AddrOperand::Space::SHARED, AddrOperand::OffsetType::REGISTER, "");
    REQUIRE(ptxsim::extract_smem_offset_placeholder(instr) == 0u);
}

TEST_CASE("extract_smem_offset_placeholder returns 0 when operands empty",
          "[unit][tcgen05][cp][helper]") {
    Tcgen05Instr instr = make_cp_instr();
    REQUIRE(ptxsim::extract_smem_offset_placeholder(instr) == 0u);
}

// ============================================================================
// processTcgen05Cp exception path tests
// ============================================================================

TEST_CASE("processTcgen05Cp throws on cta_group::2 with ADR-0018 reference",
          "[unit][tcgen05][cp][cta_group_2]") {
    TestRig rig;

    REQUIRE_THROWS_AS(
        ptxsim::processTcgen05Cp(&rig.thread(), make_cp_instr(/*cta_group=*/2)),
        UnsupportedInstructionException);

    // The exception message must name the deferred architecture decision.
    try {
        ptxsim::processTcgen05Cp(&rig.thread(), make_cp_instr(2));
        FAIL("expected UnsupportedInstructionException");
    } catch (const UnsupportedInstructionException &e) {
        const std::string what = e.what();
        REQUIRE(what.find("ADR-0018") != std::string::npos);
    }
}

TEST_CASE("processTcgen05Cp throws UnsupportedInstructionException when shared "
          "memory is null",
          "[unit][tcgen05][cp][null_smem]") {
    TestRig rig;

    // Default TestRig leaves cta().sharedMemSpace == nullptr.
    REQUIRE(rig.cta().sharedMemSpace == nullptr);

    // Per spec: missing shared memory uses UnsupportedInstructionException
    // (matches the missing-WarpContext / missing-CTAContext / cta_group::2
    // semantics — all are "environment does not support this instruction").
    REQUIRE_THROWS_AS(ptxsim::processTcgen05Cp(&rig.thread(), make_cp_instr()),
                      UnsupportedInstructionException);
}