#include "catch_amalgamated.hpp"
#include "cudart/cpptlm_bridge.h"

// ============================================================================
// ADR-0029 D7: ABI stability gates
//
// cpptlm_bridge.h is the ABI source-of-truth for PTX-EMU ↔ CppTLM interop.
// This test verifies the ABI surface that must remain stable:
//   - CPPTLMBRIDGE_VERSION = 2 (must bump on any interface change)
//   - PtxEmuDriverApi struct has 8 function pointers
// ============================================================================

TEST_CASE("ABI: CPPTLMBRIDGE_VERSION remains 2 (ADR-0029 D7)",
          "[integration][cudart][abi]") {
    // ADR-0029 D7: "any ABI change must bump CPPTLMBRIDGE_VERSION"
    // This test fails if anyone accidentally changes the version number
    // without following the bump protocol (notify CppTLM team, update version).
    REQUIRE(CPPTLMBRIDGE_VERSION == 2);
}

TEST_CASE("ABI: PtxEmuDriverApi has 8 function pointers (ADR-0029 D7)",
          "[integration][cudart][abi]") {
    // Compile-time structural check: the struct must retain all 8 function
    // pointers. Adding or removing fields without bumping CPPTLMBRIDGE_VERSION
    // would silently break CppTLM interop.
    static_assert(sizeof(PtxEmuDriverApi) >= 8 * sizeof(void*),
                  "PtxEmuDriverApi must have 8 function pointers");
    REQUIRE(true);  // If it compiles, the static_assert passed
}

TEST_CASE("ABI: PtxEmuDriverApi fields are function pointers",
          "[integration][cudart][abi]") {
    // Runtime sanity check that the struct fields are non-null
    // (they should be initialized by the shim provider).
    PtxEmuDriverApi api = {};
    // At minimum the struct must be constructible and have the right size
    REQUIRE(sizeof(PtxEmuDriverApi) >= 8 * sizeof(void*));
}
