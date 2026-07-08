// test_tcgen05_ld_parse.cpp
// =============================================================================
// Integration test (类型二): tcgen05.ld parse → IR validation.
//
// Verifies that makeTcgen05Instr constructs a Tcgen05Instr with the correct
// op_kind, qualifiers, and operands count for tcgen05.ld.
//
// Per design.md D1 (Metis 修订): 直接使用 statement_factory helper.
//
// Spec: openspec/changes/fix-tcgen05-test-coverage-gaps/specs/tcgen05-handler-test-coverage/spec.md
//   Scenario: ld parse test passes with specific assertions
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"

#include <string>
#include <vector>

TEST_CASE("tcgen05.ld parse → IR",
          "[integration][ptx][tcgen05][parse][ld]") {
    // PTX source: tcgen05.ld.sync.aligned.32x32b.shared::cta.b32 [r0], [r1];
    const std::string ptx_text =
        "tcgen05.ld.sync.aligned.32x32b.shared::cta.b32 [r0], [r1];";

    std::vector<Qualifier> qualifiers = {
        Qualifier::Q_SYNC,           // .sync
        Qualifier::Q_ALIGNED,        // .aligned
        Qualifier::Q_TCGEN05_X1,     // .32x32b (num_regs=1)
        Qualifier::Q_SHARED,         // .shared::cta (memory space)
    };

    // Build 2 operands: dst register [r0] (wrapped as RegOperand), src [r1]
    std::vector<OperandContext> operands = {
        OperandContext(RegOperand{"r", 0}),  // dst register r0
        OperandContext(RegOperand{"r", 1}),  // src register r1
    };

    auto stmt = ptxir::factory::makeTcgen05Instr(
        Tcgen05OpKind::LD, qualifiers, operands, ptx_text);

    REQUIRE(std::holds_alternative<Tcgen05Instr>(stmt.data));
    const auto& instr = std::get<Tcgen05Instr>(stmt.data);

    REQUIRE(instr.op_kind == Tcgen05OpKind::LD);
    REQUIRE(instr.qualifiers.size() == 4);
    REQUIRE(instr.qualifiers[0] == Qualifier::Q_SYNC);
    REQUIRE(instr.qualifiers[1] == Qualifier::Q_ALIGNED);
    REQUIRE(instr.qualifiers[2] == Qualifier::Q_TCGEN05_X1);
    REQUIRE(instr.qualifiers[3] == Qualifier::Q_SHARED);
    REQUIRE(instr.operands.size() == 2);
    REQUIRE(instr.instructionText == ptx_text);
}