# Task 1: Baseline (2026-06-04)

## 6 divergence integration test targets — all PASS

| # | Target | TEST_CASE | Assertions | Status |
|---|--------|-----------|------------|--------|
| 78 | integration_divergence_sync_isolated | 4 | 48 | PASS |
| 79 | integration_divergence_sync_convergence | 5 | 118 | PASS |
| 80 | integration_divergence_sync_standalone | 6 | 95 | PASS |
| 81 | integration_nested_divergence | 4 | 95 | PASS |
| 82 | integration_shortest_path_first | 4 | 12 | PASS |
| 83 | integration_post_barrier_divergence | 5 | 15 | PASS |
| **Total** | | **28** | **383** | **100% PASS** |

## Key references confirmed
- `src/ptxsim/core/AGENTS.md:48-49` — known issue documentation
- `src/ptxsim/core/sm_context.cpp:536-637` — `synchronize_barrier()` (no update_active_mask call)
- `tests/integration/CMakeLists.txt:77-105` — 6 divergence test targets
- `scripts/sanity.sh:175-176` — regex matching `test_post_barrier_divergence` and `test_divergence_sync_standalone|test_divergence_sync_isolated`
- `AGENTS.md:207` — documents `integration_divergence_sync_isolated`, `integration_divergence_sync_convergence` in divergence label
- `docs/adr/0013-statement-factory-test-unification.md` — references `test_divergence_sync_isolated.cpp` and `test_post_barrier_divergence.cpp`
- `docs/testing/TEST_DOCUMENTATION.md:217` — `test_post_barrier_divergence.cpp` (5 TEST_CASE)
- `workflow-state.md:85` — `test_shortest_path_first.cpp`
- `.opencode/skills/ptx-lane-verification/SKILL.md:446-448` — references iso and nested files

## ctest label filter note
`ctest -L "integration;divergence"` returns "No tests were found" on this build.
Use `ctest -R "integration_(divergence|nested|shortest|post_barrier)"` instead.
