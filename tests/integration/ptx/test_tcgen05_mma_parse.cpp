// test_tcgen05_mma_parse.cpp
// =============================================================================
// Integration test (类型二): tcgen05.mma parse → IR validation.
//
// Verifies that makeTcgen05Instr constructs a Tcgen05Instr with the correct
// op_kind, qualifiers, and operands count for the tcgen05.mma instruction.
//
// Per design.md D1 (Metis 修订): 直接使用 statement_factory helper
// (无 ANTLR 解析路径 — 测试代码不应驱动实际 ANTLR parser).
//
// Spec: openspec/changes/fix-tcgen05-test-coverage-gaps/specs/tcgen05-handler-test-coverage/spec.md
//   Requirement: "5 integration parse tests SHALL exist for 5 core handlers"
//   Scenario: mma parse test passes with specific assertions
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"

#include <string>
#include <vector>

TEST_CASE("tcgen05.mma parse → IR",
          "[integration][ptx][tcgen05][parse][mma]") {
    // PTX source: tcgen05.mma.kind::f16.cta_group::1 d, a, b, c;
    const std::string ptx_text =
        "tcgen05.mma.kind::f16.cta_group::1 d, a, b, c;";

    // Build qualifiers (KIND::F16 + CTA_GROUP::1)
    std::vector<ptxemu::ir::Qualifier> qualifiers = {
        ptxemu::ir::Qualifier::Q_F16,              // .kind::f16
        ptxemu::ir::Qualifier::Q_TCGEN_CTA_GROUP,  // .cta_group::1
    };

    // Build 4 operands (d, a, b, c) — placeholder register operands
    std::vector<ptxemu::ir::OperandContext> operands = {
        ptxemu::ir::OperandContext(RegOperand{"d", 0}),
        ptxemu::ir::OperandContext(RegOperand{"a", 0}),
        ptxemu::ir::OperandContext(RegOperand{"b", 0}),
        ptxemu::ir::OperandContext(RegOperand{"c", 0}),
    };

    auto stmt = ptxir::factory::makeTcgen05Instr(
        ptxemu::ir::Tcgen05OpKind::MMA, qualifiers, operands, ptx_text);

    // Verify: IR carries Tcgen05Instr
    REQUIRE(std::holds_alternative<ptxemu::ir::Tcgen05Instr>(stmt.data));
    const auto& instr = std::get<ptxemu::ir::Tcgen05Instr>(stmt.data);

    // Verify: op_kind
    REQUIRE(instr.op_kind == ptxemu::ir::Tcgen05OpKind::MMA);

    // Verify: qualifiers (KIND::F16 + CTA_GROUP)
    REQUIRE(instr.qualifiers.size() == 2);
    REQUIRE(instr.qualifiers[0] == ptxemu::ir::Qualifier::Q_F16);
    REQUIRE(instr.qualifiers[1] == ptxemu::ir::Qualifier::Q_TCGEN_CTA_GROUP);

    // Verify: operands count == 4
    REQUIRE(instr.operands.size() == 4);

    // Verify: instruction text retained
    REQUIRE(instr.instructionText == ptx_text);
}

TEST_CASE("tcgen05.mma.cta_group::2 populates instr.cta_group (FU-1 C3)",
          "[integration][ptx][tcgen05][parse][cta_group][FU-1]") {
    // FU-1 Oracle C3 fix: verify that makeTcgen05Instr with cta_group=2
    // correctly stores the value. The visitor's IMMEDIATE extraction
    // (ptx_visitor.cpp:visitTcgen05Inst) will be verified separately via
    // full PTX parse path tests.
    const std::string ptx_text =
        "tcgen05.mma.kind::f16.cta_group::2 d, a, b, c;";

    std::vector<ptxemu::ir::Qualifier> qualifiers = {
        ptxemu::ir::Qualifier::Q_F16,
        ptxemu::ir::Qualifier::Q_TCGEN_CTA_GROUP,
    };

    std::vector<ptxemu::ir::OperandContext> operands = {
        ptxemu::ir::OperandContext(RegOperand{"d", 0}),
        ptxemu::ir::OperandContext(RegOperand{"a", 0}),
        ptxemu::ir::OperandContext(RegOperand{"b", 0}),
        ptxemu::ir::OperandContext(RegOperand{"c", 0}),
    };

    auto stmt = ptxir::factory::makeTcgen05Instr(
        ptxemu::ir::Tcgen05OpKind::MMA, qualifiers, operands, ptx_text, /*cta_group=*/2);

    REQUIRE(std::holds_alternative<ptxemu::ir::Tcgen05Instr>(stmt.data));
    const auto& instr = std::get<ptxemu::ir::Tcgen05Instr>(stmt.data);

    // Verify cta_group was stored (this FAILS pre-fix because old
    // makeTcgen05Instr had no cta_group parameter)
    REQUIRE(instr.cta_group == 2u);

    // Verify Q_TCGEN_CTA_GROUP is in the qualifier list
    REQUIRE(std::find(instr.qualifiers.begin(), instr.qualifiers.end(),
                      ptxemu::ir::Qualifier::Q_TCGEN_CTA_GROUP) != instr.qualifiers.end());
}