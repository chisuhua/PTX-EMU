## ADDED Requirements

### Requirement: CppTLM cpptlm_core enables PIC

The system SHALL configure the CppTLM `ExternalProject_Add` to build `cpptlm_core` with position-independent code.

- SHALL pass `-DCMAKE_POSITION_INDEPENDENT_CODE=ON` to CppTLM CMake configure step
- SHALL ensure `cpptlm_core` static library can be linked into shared libraries (`.so`)

### Requirement: PTX-EMU pins CppTLM commit hash

The system SHALL pin `CPPTLM_COMMIT_HASH` to a specific verified commit for reproducible builds.

- SHALL set `CPPTLM_COMMIT_HASH` to `73e5422` (verified P0+P1 Phase 1 commit)
- SHALL use `CACHE STRING` for overridability
- SHALL default to the pinned hash, not `"main"`

#### Scenario: ExternalProject_Add uses pinned commit

- **WHEN** CMake configures with `BUILD_LIB_CPPTLM_CUDART=ON`
- **THEN** ExternalProject_Add for CppTLM SHALL use `GIT_TAG 73e5422`
- **THEN** the CppTLM header files SHALL include `IPtxEmuDriver` interface and 3 vendor headers

#### Scenario: cpptlm_bridge subdirectory builds

- **WHEN** `BUILD_LIB_CPPTLM_CUDART=ON`
- **THEN** `src/cudart/cpptlm_bridge/CMakeLists.txt` SHALL be included in the build
- **THEN** `PtxEmuDriverShim` SHALL be compiled and linked into `cudart` target