// ============================================================================
// DEAD-CODE COVERAGE TEST — see design.md D4
//
// processTcgen05Mma is currently NOT routed via dispatch:
//   - S_TCGEN05_* in ptx_op.def:129-136 is explicitly excluded from X-Macro
//   - InstructionFactory::get_handler(S_TCGEN05_MMA) returns nullptr
//   - ThreadContext::execute_thread_instruction() walks through
//     "No handler found" path (thread_context.cpp:142-146) and set_state(EXIT)
//
// This test:
//   1. Validates the golden value file compiles + has correct size
//   2. Verifies specific hand-computed values (per spec.md Scenario 2)
//   3. Validates tcgen05.h declares processTcgen05Mma (compile-time check)
//
// When the dispatch path is fixed by `fix-tcgen05-handler-dispatch`, an
// additional integration test will drive real warp execution and compare
// TMEM output against the same golden values. The dispatch path itself
// cannot be tested in this PR (handler is dead code).
//
// Note on direct invocation: calling processTcgen05Mma requires a fully
// initialized CTAContext + WarpContext + ThreadContext hierarchy (via
// CTAContext::init), which is heavyweight for a unit test. The IR
// construction path is exercised by the 5 integration parse tests in
// tests/integration/ptx/test_tcgen05_*_parse.cpp.
// ============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/instructions/tcgen05.h"
#include "reference/ptx_tcgen05/tcgen05_mma_golden.h"

#include <array>
#include <cstdint>

using ptxsim::reference::tcgen05::GOLDEN_MMA_F16_F16_F32;

TEST_CASE("GOLDEN_MMA_F16_F16_F32 has 32 elements (8 rows × 4 cols)",
          "[unit][ptx_ir][tcgen05][mma][golden][size]") {
    REQUIRE(GOLDEN_MMA_F16_F16_F32.size() == 32);
}

TEST_CASE("GOLDEN_MMA_F16_F16_F32 contains hand-verifiable values",
          "[unit][ptx_ir][tcgen05][mma][golden][values]") {
    // Per spec.md Scenario 2: each value = A[i/4] * B[i%4], A=[1..8], B=[1..4]
    // C[i][j] = A[i] * B[j]; index = i*4 + j
    REQUIRE(GOLDEN_MMA_F16_F16_F32[0]  == 1.0f);   // C[0][0] = 1*1
    REQUIRE(GOLDEN_MMA_F16_F16_F32[3]  == 4.0f);   // C[0][3] = 1*4
    REQUIRE(GOLDEN_MMA_F16_F16_F32[4]  == 2.0f);   // C[1][0] = 2*1
    REQUIRE(GOLDEN_MMA_F16_F16_F32[7]  == 8.0f);   // C[1][3] = 2*4
    REQUIRE(GOLDEN_MMA_F16_F16_F32[28] == 8.0f);   // C[7][0] = 8*1
    REQUIRE(GOLDEN_MMA_F16_F16_F32[31] == 32.0f);  // C[7][3] = 8*4

    // Verify the entire array is the expected product pattern
    constexpr int ROWS = 8;
    constexpr int COLS = 4;
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            const size_t idx = static_cast<size_t>(i * COLS + j);
            const float a = static_cast<float>(i + 1);  // A[i] = i+1
            const float b = static_cast<float>(j + 1);  // B[j] = j+1
            INFO("i=" << i << " j=" << j << " idx=" << idx
                 << " expected=" << a * b
                 << " actual=" << GOLDEN_MMA_F16_F16_F32[idx]);
            REQUIRE(GOLDEN_MMA_F16_F16_F32[idx] == Catch::Approx(a * b));
        }
    }
}

TEST_CASE("tcgen05.h forward declaration compiles and links",
          "[unit][ptx_ir][tcgen05][mma][golden][link]") {
    // Compile-time check: if this test file compiles, the forward
    // declarations are valid (we #include the header above).
    //
    // We do NOT call processTcgen05Mma here because:
    //   1. processTcgen05Mma accesses cta->tmem() which requires a fully
    //      initialized CTAContext (via CTAContext::init)
    //   2. cta is reached via context->get_warp_context()->get_cta_context()
    //      which requires WarpContext parent linkage
    //   3. Constructing all three contexts is heavyweight and orthogonal
    //      to the dead-code coverage purpose (dispatch is broken)
    //
    // Direct invocation will be exercised by integration tests once the
    // dispatch is wired by fix-tcgen05-handler-dispatch.
    //
    // Taking the address of the function confirms the symbol exists.
    using FnPtr = void (*)(ThreadContext*, const Tcgen05Instr&);
    FnPtr fn = &ptxsim::processTcgen05Mma;
    REQUIRE(fn != nullptr);
}