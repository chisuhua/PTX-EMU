# cmake-antlr4-relative-paths spec

## ADDED Requirements

### Requirement: PTX-EMU `CMakeLists.txt` MUST use `PROJECT_SOURCE_DIR` for vendored ANTLR4 paths

The `CMakeLists.txt` MUST use `${PROJECT_SOURCE_DIR}` (not `${CMAKE_SOURCE_DIR}`) when referencing vendored ANTLR4 paths:

- `ANTLR_EXECUTABLE`: path to `antlr-4.13.2-complete.jar`
- `ANTLR4_RUNTIME_SOURCE_DIR`: path to `antlr4-cpp-runtime-4.13.2-source/`

The `PROJECT_SOURCE_DIR` variable resolves to the directory where the most recent `project()` command was called — i.e., PTX-EMU's own root. This guarantees correct ANTLR4 path resolution when PTX-EMU is consumed as a subproject via `add_subdirectory(external/PTX-EMU)` or `ExternalProject_Add(...)` from CppTLM (or any other parent CMake project).

#### Scenario: Standalone PTX-EMU build succeeds

- **WHEN** `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` is run from PTX-EMU's root directory
- **THEN** CMake configuration succeeds
- **AND** ANTLR4 paths resolve correctly (zero `file(STRINGS ...)` or `add_custom_command(... WORKING_DIRECTORY ...)` errors)
- **AND** the build proceeds identically to pre-fix behavior

#### Scenario: CppTLM-side `add_subdirectory` consumption succeeds

- **WHEN** a parent project (e.g., CppTLM) writes:
  ```cmake
  cmake_minimum_required(VERSION 3.18)
  project(CppTLM LANGUAGES CXX)
  add_subdirectory(external/PTX-EMU)
  ```
- **THEN** PTX-EMU's `CMakeLists.txt` processes correctly
- **AND** ANTLR4 paths resolve to PTX-EMU's own `${PROJECT_SOURCE_DIR}/antlr4/` (NOT to CppTLM's `${CMAKE_SOURCE_DIR}/antlr4/`)
- **AND** zero ANTLR4 path-related CMake errors

> **DEFERRED to HSK-9**: This scenario is verified by `tests/build_cpptlm_consume/consumer_smoke` (HSK-9 entry). For this change (`antlr4-path-hardcoding-fix`), the scenario is verified by **static analysis** + `CMAKE` documentation reference that `${PROJECT_SOURCE_DIR}` is the correct CMake variable for project-relative paths. Standalone PTX-EMU build (Scenarios 1 + 4) is the executable verification for this change.

#### Scenario: CppTLM-side `ExternalProject_Add` consumption succeeds

- **WHEN** a parent project uses `ExternalProject_Add(... PTX-EMU GIT_REPOSITORY ...)` to fetch and build PTX-EMU
- **THEN** PTX-EMU's `CMakeLists.txt` processes correctly inside the ExternalProject subdirectory
- **AND** ANTLR4 paths resolve to the cloned PTX-EMU subdirectory's `${PROJECT_SOURCE_DIR}/antlr4/`

> **DEFERRED to HSK-9**: Same as Scenario 2. Verification via `consumer_smoke` test in HSK-9 entry. This change provides only the static guarantee (correct variable usage).

#### Scenario: Regression to `${CMAKE_SOURCE_DIR}/antlr4` is blocked by drift_check

- **WHEN** a future commit modifies `CMakeLists.txt` to re-introduce `${CMAKE_SOURCE_DIR}/antlr4` hardcoded path
- **THEN** drift_check workflow Invariant 7 FAILS
- **AND** the CI pipeline blocks merge to main
- **AND** the regression is detected before reaching CppTLM-side chained builds

#### Scenario: drift_check Invariant 7 implementation

- **MATCH**: `^\s*set\s*\(\s*ANTLR(4)?_(EXECUTABLE|RUNTIME_SOURCE_DIR)\s+\$\{CMAKE_SOURCE_DIR\}/antlr4`
- **EXCLUDE**: `${CMAKE_CURRENT_SOURCE_DIR}/antlr4` (acceptable for subdirectory-relative references)
- **EXCLUDE**: `${PROJECT_SOURCE_DIR}/antlr4` (the correct fix)
- **IMPLEMENTATION**: bash + grep -nE "CMAKE_SOURCE_DIR.*antlr4" CMakeLists.txt (returns 0 lines = PASS)