// test_tcgen05_st_parse.cpp
// =============================================================================
// Integration test (类型二): tcgen05.st parse → IR validation.
//
// Verifies that makeTcgen05Instr constructs a Tcgen05Instr with the correct
// op_kind, qualifiers, and operands count for tcgen05.st (symmetric to ld).
//
// Per design.md D1 (Metis 修订): 直接使用 statement_factory helper.
//
// Spec: openspec/changes/fix-tcgen05-test-coverage-gaps/specs/tcgen05-handler-test-coverage/spec.md
//   Scenario: st parse test passes (symmetric to ld)
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"

#include <string>
#include <vector>

TEST_CASE("tcgen05.st parse → IR",
          "[integration][ptx][tcgen05][parse][st]") {
    // PTX source: tcgen05.st.sync.aligned.32x32b.shared::cta.b32 [r0], [r1];
    const std::string ptx_text =
        "tcgen05.st.sync.aligned.32x32b.shared::cta.b32 [r0], [r1];";

    std::vector<ptxemu::ir::Qualifier> qualifiers = {
        ptxemu::ir::Qualifier::Q_SYNC,           // .sync
        ptxemu::ir::Qualifier::Q_ALIGNED,        // .aligned
        ptxemu::ir::Qualifier::Q_TCGEN05_X1,     // .32x32b (num_regs=1)
        ptxemu::ir::Qualifier::Q_SHARED,         // .shared::cta (memory space)
    };

    // 2 operands: dst_addr [r0] + src [r1] (note: st operand order is reversed vs ld)
    std::vector<ptxemu::ir::OperandContext> operands = {
        ptxemu::ir::OperandContext(RegOperand{"r", 0}),  // dst address
        ptxemu::ir::OperandContext(RegOperand{"r", 1}),  // src register
    };

    auto stmt = ptxir::factory::makeTcgen05Instr(
        ptxemu::ir::Tcgen05OpKind::ST, qualifiers, operands, ptx_text);

    REQUIRE(std::holds_alternative<ptxemu::ir::Tcgen05Instr>(stmt.data));
    const auto& instr = std::get<ptxemu::ir::Tcgen05Instr>(stmt.data);

    REQUIRE(instr.op_kind == ptxemu::ir::Tcgen05OpKind::ST);
    REQUIRE(instr.qualifiers.size() == 4);
    REQUIRE(instr.qualifiers[0] == ptxemu::ir::Qualifier::Q_SYNC);
    REQUIRE(instr.qualifiers[1] == ptxemu::ir::Qualifier::Q_ALIGNED);
    REQUIRE(instr.qualifiers[2] == ptxemu::ir::Qualifier::Q_TCGEN05_X1);
    REQUIRE(instr.qualifiers[3] == ptxemu::ir::Qualifier::Q_SHARED);
    REQUIRE(instr.operands.size() == 2);
    REQUIRE(instr.instructionText == ptx_text);
}