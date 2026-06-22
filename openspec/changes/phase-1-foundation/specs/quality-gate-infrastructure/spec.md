## ADDED Requirements

### Requirement: compile_commands.json generation
The build system MUST generate `build/compile_commands.json` so that language servers (LSP), `clang-tidy`, IDEs (clangd/VS Code), and other tooling can resolve source file paths and compile flags.

#### Scenario: Clean build produces compile_commands.json
- **WHEN** developer runs `cmake -S . -B build` on a clean checkout
- **THEN** the file `build/compile_commands.json` MUST exist and contain a non-empty JSON array of compilation commands for every translation unit in the project

#### Scenario: clangd can parse the generated file
- **WHEN** developer opens a source file (e.g., `src/ptxsim/core/thread_context.cpp`) in their LSP-enabled editor
- **THEN** the LSP MUST resolve header paths, symbol references, and provide go-to-definition without errors

#### Scenario: Stale or broken symlink at project root is removed
- **WHEN** `compile_commands.json` exists at the project root as a broken symlink pointing to a non-existent build directory
- **THEN** the symlink MUST be removed (build system regenerates the file at `build/compile_commands.json` instead)

### Requirement: Continuous integration workflow
The repository MUST contain a GitHub Actions workflow `.github/workflows/build-test.yml` that builds the project and runs the test suite on every pull request and push to `main`.

#### Scenario: Pull request triggers build and test
- **WHEN** a developer opens a pull request targeting `main`
- **THEN** the `build-test` workflow MUST execute automatically and report pass/fail status on the PR

#### Scenario: Workflow installs correct CUDA Toolkit version
- **WHEN** the workflow runner initializes
- **THEN** it MUST install CUDA Toolkit `11.4.4` (matching the project's documented version in `README.md`) via `Jimver/cuda-toolkit` GitHub Action

#### Scenario: Workflow builds project using Ninja generator
- **WHEN** the workflow runs the configure step
- **THEN** it MUST invoke `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release` for faster incremental builds

#### Scenario: Workflow runs the test suite with disabled tests excluded
- **WHEN** the workflow runs the test step
- **THEN** it MUST invoke `cd build && ctest --output-on-failure -E "Disabled"` so that tests marked as Disabled do not fail the workflow

### Requirement: Regression baseline archive
The repository MUST archive a regression baseline `docs/audits/baseline-2026-06-21.log` containing the output of `./scripts/sanity.sh` plus a summary of ctest test states (pass / fail / disabled), so that future regressions can be compared against an objective snapshot.

#### Scenario: Baseline captures full sanity output
- **WHEN** developer runs `./scripts/sanity.sh 2>&1 | tee docs/audits/baseline-2026-06-21.log`
- **THEN** the file MUST contain the complete stdout and stderr of the sanity script including build warnings, test pass/fail, and disabled test markers

#### Scenario: Baseline includes ctest state summary
- **WHEN** developer runs `ctest -N` after the sanity script
- **THEN** the output MUST be appended to the baseline log so the file includes a per-test enumerated list with pass/fail/disabled status

#### Scenario: Baseline enables diff-based regression detection
- **WHEN** a future commit causes a test to fail that passed in the baseline
- **THEN** a `diff baseline-2026-06-21.log current-sanity.log` MUST show the regression as a state transition (PASS → FAIL)

### Requirement: xfail policy for first-time CI failures
When CI is first enabled, any test that fails on the baseline MUST be marked as `xfail` (expected to fail) rather than blocking the pull request, so that existing technical debt does not stall all future merges.

#### Scenario: Baseline failure does not block merge
- **WHEN** a test fails on the baseline and is marked `xfail`
- **THEN** the pull request MUST be mergeable despite the failure, with a separate tracking issue created per failure

#### Scenario: xfail tracking issue per failure
- **WHEN** a test is marked `xfail`
- **THEN** a corresponding GitHub issue MUST be opened documenting the failure root cause and linking to the fix plan (typically a future roadmap task)