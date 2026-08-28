// test_tcgen05_wait_parse.cpp
// =============================================================================
// Integration test (类型二): tcgen05.wait parse → IR validation (load + store variants).
//
// Verifies zero-operand variant: makeTcgen05Instr constructs a Tcgen05Instr
// with op_kind == WAIT, cta_group qualifier, and empty operands.
//
// Per design.md D1 (Metis 修订): 直接使用 statement_factory helper.
//
// Note: tcgen05.wait::load / .store distinction is sub-op specific to WAIT
// (op_kind already = WAIT). Both variants share the same Tcgen05Instr layout.
//
// Spec: openspec/changes/fix-tcgen05-test-coverage-gaps/specs/tcgen05-handler-test-coverage/spec.md
//   Scenario: wait parse test passes (load + store variants)
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"

#include <string>
#include <vector>

TEST_CASE("tcgen05.wait::load parse → IR",
          "[integration][ptx][tcgen05][parse][wait][load]") {
    // PTX source: tcgen05.wait::load.cta_group::1;
    const std::string ptx_text = "tcgen05.wait::load.cta_group::1;";

    std::vector<ptxemu::ir::Qualifier> qualifiers = {
        ptxemu::ir::Qualifier::Q_TCGEN_CTA_GROUP,  // .cta_group::1
        ptxemu::ir::Qualifier::Q_TCGEN05_WAIT,     // .wait marker
    };
    std::vector<ptxemu::ir::OperandContext> operands;  // wait is zero-operand

    auto stmt = ptxir::factory::makeTcgen05Instr(
        ptxemu::ir::Tcgen05OpKind::WAIT, qualifiers, operands, ptx_text);

    REQUIRE(std::holds_alternative<ptxemu::ir::Tcgen05Instr>(stmt.data));
    const auto& instr = std::get<ptxemu::ir::Tcgen05Instr>(stmt.data);

    REQUIRE(instr.op_kind == ptxemu::ir::Tcgen05OpKind::WAIT);
    REQUIRE(instr.operands.size() == 0);
    REQUIRE(instr.qualifiers.size() == 2);
    REQUIRE(instr.qualifiers[0] == ptxemu::ir::Qualifier::Q_TCGEN_CTA_GROUP);
    REQUIRE(instr.qualifiers[1] == ptxemu::ir::Qualifier::Q_TCGEN05_WAIT);
    REQUIRE(instr.instructionText == ptx_text);
}

TEST_CASE("tcgen05.wait::store parse → IR",
          "[integration][ptx][tcgen05][parse][wait][store]") {
    // PTX source: tcgen05.wait::store.cta_group::1;
    const std::string ptx_text = "tcgen05.wait::store.cta_group::1;";

    std::vector<ptxemu::ir::Qualifier> qualifiers = {
        ptxemu::ir::Qualifier::Q_TCGEN_CTA_GROUP,  // .cta_group::1
        ptxemu::ir::Qualifier::Q_TCGEN05_WAIT,     // .wait marker
    };
    std::vector<ptxemu::ir::OperandContext> operands;  // wait is zero-operand

    auto stmt = ptxir::factory::makeTcgen05Instr(
        ptxemu::ir::Tcgen05OpKind::WAIT, qualifiers, operands, ptx_text);

    REQUIRE(std::holds_alternative<ptxemu::ir::Tcgen05Instr>(stmt.data));
    const auto& instr = std::get<ptxemu::ir::Tcgen05Instr>(stmt.data);

    REQUIRE(instr.op_kind == ptxemu::ir::Tcgen05OpKind::WAIT);
    REQUIRE(instr.operands.size() == 0);
    REQUIRE(instr.qualifiers.size() == 2);
    REQUIRE(instr.qualifiers[0] == ptxemu::ir::Qualifier::Q_TCGEN_CTA_GROUP);
    REQUIRE(instr.qualifiers[1] == ptxemu::ir::Qualifier::Q_TCGEN05_WAIT);
    REQUIRE(instr.instructionText == ptx_text);
}