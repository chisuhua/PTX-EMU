/**
 * PTX-6 TDD Red-Phase: tests for 3 static helper methods on SMContext
 *
 * Tests verify the helper functions in SMContext::exe_once() 3-step injection
 * (Phase 8.B PTX-6). These are public static methods to enable unit testing
 * without requiring a full SMContext + WarpContext setup.
 *
 * Phase 8.B PTX-6 — cpptlm-phase8b-injection-points
 * Ref: ADR-0020, design.md §7.2, Oracle review 2026-07-17
 */
#include "catch_amalgamated.hpp"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/sm_context.h"

#include <map>
#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// Test #1: is_tensor_core_instruction() — X-Macro enum range verification
// ---------------------------------------------------------------------------
TEST_CASE("is_tensor_core_instruction: S_TCGEN05_ALLOC..FENCE are TC",
          "[unit][sm][ptx6][helper]") {
    StatementContext stmt;

    // Test 11 tcgen05 instructions are detected as TC
    stmt.type = S_TCGEN05_ALLOC;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == true);

    stmt.type = S_TCGEN05_DEALLOC;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == true);

    stmt.type = S_TCGEN05_RELINQUISH;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == true);

    stmt.type = S_TCGEN05_LD;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == true);

    stmt.type = S_TCGEN05_ST;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == true);

    stmt.type = S_TCGEN05_CP;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == true);

    stmt.type = S_TCGEN05_MMA;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == true);

    stmt.type = S_TCGEN05_MMA_WS;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == true);

    stmt.type = S_TCGEN05_COMMIT;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == true);

    stmt.type = S_TCGEN05_WAIT;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == true);

    stmt.type = S_TCGEN05_FENCE;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == true);
}

TEST_CASE("is_tensor_core_instruction: arithmetic NOT TC",
          "[unit][sm][ptx6][helper]") {
    StatementContext stmt;

    stmt.type = S_ADD;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == false);

    stmt.type = S_MUL;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == false);

    stmt.type = S_LD;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == false);

    stmt.type = S_ST;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == false);

    stmt.type = S_BRA;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == false);

    stmt.type = S_BAR;
    REQUIRE(SMContext::is_tensor_core_instruction(stmt) == false);
}

// ---------------------------------------------------------------------------
// Test #2: map_instruction_to_tc_precision() — qualifier → TcPrecision
// ---------------------------------------------------------------------------
TEST_CASE("map_instruction_to_tc_precision: .f16 qualifier → FP16",
          "[unit][sm][ptx6][helper]") {
    StatementContext stmt;
    GenericInstr instr;
    instr.qualifiers.push_back(Qualifier::Q_F16);
    stmt.data = instr;

    REQUIRE(SMContext::map_instruction_to_tc_precision(stmt) == TcPrecision::FP16);
}

TEST_CASE("map_instruction_to_tc_precision: .bf16 qualifier → BF16",
          "[unit][sm][ptx6][helper]") {
    StatementContext stmt;
    GenericInstr instr;
    instr.qualifiers.push_back(Qualifier::Q_BF16);
    stmt.data = instr;

    REQUIRE(SMContext::map_instruction_to_tc_precision(stmt) == TcPrecision::BF16);
}

TEST_CASE("map_instruction_to_tc_precision: .tf32 qualifier → TF32",
          "[unit][sm][ptx6][helper]") {
    StatementContext stmt;
    GenericInstr instr;
    instr.qualifiers.push_back(Qualifier::Q_TCGEN_TF32);
    stmt.data = instr;

    REQUIRE(SMContext::map_instruction_to_tc_precision(stmt) == TcPrecision::TF32);
}

TEST_CASE("map_instruction_to_tc_precision: no qualifier → fallback FP16",
          "[unit][sm][ptx6][helper]") {
    StatementContext stmt;
    GenericInstr instr;
    // No qualifiers
    stmt.data = instr;

    REQUIRE(SMContext::map_instruction_to_tc_precision(stmt) == TcPrecision::FP16);
}

// ---------------------------------------------------------------------------
// Test #3: map_instruction_to_pipeline() — stmt.type → PipelineId
// ---------------------------------------------------------------------------
TEST_CASE("map_instruction_to_pipeline: arithmetic → P0_INT_FP32",
          "[unit][sm][ptx6][helper]") {
    StatementContext stmt;

    stmt.type = S_ADD;
    REQUIRE(SMContext::map_instruction_to_pipeline(stmt) == PipelineId::P0_INT_FP32);

    stmt.type = S_MUL;
    REQUIRE(SMContext::map_instruction_to_pipeline(stmt) == PipelineId::P0_INT_FP32);
}

TEST_CASE("map_instruction_to_pipeline: ld/st → P3_LSU",
          "[unit][sm][ptx6][helper]") {
    StatementContext stmt;

    stmt.type = S_LD;
    REQUIRE(SMContext::map_instruction_to_pipeline(stmt) == PipelineId::P3_LSU);

    stmt.type = S_ST;
    REQUIRE(SMContext::map_instruction_to_pipeline(stmt) == PipelineId::P3_LSU);
}

TEST_CASE("map_instruction_to_pipeline: tcgen05 → P4_TC",
          "[unit][sm][ptx6][helper]") {
    StatementContext stmt;

    stmt.type = S_TCGEN05_MMA;
    REQUIRE(SMContext::map_instruction_to_pipeline(stmt) == PipelineId::P4_TC);

    stmt.type = S_TCGEN05_LD;
    REQUIRE(SMContext::map_instruction_to_pipeline(stmt) == PipelineId::P4_TC);
}
