// test_tcgen05_commit_parse.cpp
// =============================================================================
// Integration test (类型二): tcgen05.commit parse → IR validation.
//
// Verifies zero-operand variant: makeTcgen05Instr constructs a Tcgen05Instr
// with op_kind == COMMIT, cta_group qualifier, and empty operands.
//
// Per design.md D1 (Metis 修订): 直接使用 statement_factory helper.
//
// Spec: openspec/changes/fix-tcgen05-test-coverage-gaps/specs/tcgen05-handler-test-coverage/spec.md
//   Scenario: commit parse test passes (zero-operand variant)
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"

#include <string>
#include <vector>

TEST_CASE("tcgen05.commit parse → IR",
          "[integration][ptx][tcgen05][parse][commit]") {
    // PTX source: tcgen05.commit.cta_group::1;
    const std::string ptx_text = "tcgen05.commit.cta_group::1;";

    std::vector<ptxemu::ir::Qualifier> qualifiers = {
        ptxemu::ir::Qualifier::Q_TCGEN_CTA_GROUP,  // .cta_group::1
    };

    // Zero operands (commit is a sync-only instruction)
    std::vector<ptxemu::ir::OperandContext> operands;

    auto stmt = ptxir::factory::makeTcgen05Instr(
        ptxemu::ir::Tcgen05OpKind::COMMIT, qualifiers, operands, ptx_text);

    REQUIRE(std::holds_alternative<ptxemu::ir::Tcgen05Instr>(stmt.data));
    const auto& instr = std::get<ptxemu::ir::Tcgen05Instr>(stmt.data);

    REQUIRE(instr.op_kind == ptxemu::ir::Tcgen05OpKind::COMMIT);
    REQUIRE(instr.qualifiers.size() == 1);
    REQUIRE(instr.qualifiers[0] == ptxemu::ir::Qualifier::Q_TCGEN_CTA_GROUP);
    REQUIRE(instr.operands.size() == 0);
    REQUIRE(instr.instructionText == ptx_text);
}