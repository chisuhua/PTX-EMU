## Why

`tests/integration/divergence/test_nested_divergence.cpp:106` carries an open
TODO declaring that the file's current `setp+selp` scenario is **not** a real
nested-divergence test — it does not push the SIMT stack twice. The audit
(`docs/audits/debt-audit-2026-07-02.md` §2.1 P1-A10) flags this missing
two-level `@%p bra` coverage as the last remaining P1 architecture debt after
A-9 atomic-CAS was archived 2026-07-06. Without this test, regressions in
nested `handle_branch` (SIMT stack push/pop, post-dominator reconvergence)
would be caught by manual replays, not the regression suite.

This change closes A-10 by adding a true two-level `@%p bra` test that drives
a 32-lane warp through an outer + inner divergent branch and verifies that
every lane ends up at the unified convergence PC with the right per-lane
register values.

## What Changes

- Add a new TEST_CASE to `tests/integration/divergence/test_nested_divergence.cpp`
  that exercises two nested `@%p bra` instructions against a 32-lane warp.
- The test:
  - Outer divergence (`@p1 bra`): lanes with `p1=true` enter the then-block
    (set `r1 = 100`); lanes with `p1=false` fall through to `r1 = 1000`.
  - Inner divergence (`@p2 bra` inside the then-block): among the lanes that
    took the then-block, those with `p2=true` set `r2 = 200`; others set
    `r2 = 300`.
  - Convergence: every lane executes a unifying `add` and reaches `ret`
    regardless of which path it walked.
  - Verifications:
    - SIMT stack depth grows to 2 during the inner branch (per
      `tests/integration/divergence/test_post_barrier_reconvergence_simplegemm.cpp`
      pattern) and returns to 0 after the final `bra` reconverges.
    - Per-lane register values match the expected path table (8 lanes:
      `r1=100, r2=200`; 8 lanes: `r1=100, r2=300`; 16 lanes: `r1=1000`).
    - All 32 lanes reach the unified convergence PC.

## Capabilities

### New Capabilities
- `nested-divergence-coverage`: regression coverage for two-level
  `@%p bra` divergence, exercising SIMT stack push/pop and post-dominator
  reconvergence end-to-end.

### Modified Capabilities
- None

## Impact

### Affected code
- `tests/integration/divergence/test_nested_divergence.cpp` — add one
  TEST_CASE (and any helpers) without removing the existing
  `setp+selp` scenario (preserves historical behavior; the existing test
  verifies predicate-driven value selection which is complementary to
  real nested branching).

### Not affected
- The SIMT stack implementation (`src/ptxsim/core/simt_stack.{h,cpp}`),
  `WarpContext::handle_branch` (`src/ptxsim/core/warp_context.cpp`), and
  any barrier/divergence logic. This change adds **coverage**; no
  correctness fixes are anticipated.
- PTX parser, runtime, AGENTS.md (only the existing 2026-05-08 file
  header comment gets retired from "this file uses setp/selp" to
  "complementary setp+selp and @%p bra tests coexist"; the original
  comment is preserved verbatim at the top of the file).

## Design-Time Checklist (Lessons-Learned)

### Function migration
- [N/A] This change is test-only. No handler code is touched.

### Multi-Phase 推进
- [N/A] Single Phase (test authoring only). No baseline-worktree
  requirement; existing 14-commit-ahead main has all dependencies.

### 文档同步
- [x] Tests/AGENTS.md already lists
  `tests/integration/divergence/` as the canonical location for divergence
  integration tests; no entry to add.
- [x] `docs/roadmap/post-phase3-debt-roadmap.md §1.1 A-10` will be
  updated to RESOLVED on archive.

### Scope lock
- [x] Adding **only** a new TEST_CASE; not modifying
  `test_divergence_sync_*` files or the existing `setp+selp` case.
- [x] No anti-patterns introduced: no `force_set_pc`, no direct
  `set_active_mask` write, no `qualifiers.back()` (per
  `ptx-lessons-learned` §1 §2 §5 + AGENTS.md §ANTI-PATTERNS).

## Refs

- Debt audit: `docs/audits/debt-audit-2026-07-02.md` §2.1 P1-A10
- Roadmap: `docs/roadmap/post-phase3-debt-roadmap.md` §1.1 A-10
- File with TODO: `tests/integration/divergence/test_nested_divergence.cpp:106`
- SIMT stack discipline: AGENTS.md §OVERVIEW + `src/ptxsim/core/AGENTS.md`
  DUAL STATE MECHANISM (per-thread PC + warp_state.threads[i].pc)
- Reference patterns (read-only):
  - `tests/integration/divergence/test_post_barrier_reconvergence_simplegemm.cpp`
  - `tests/integration/divergence/test_divergence_sync_convergence.cpp`
- Helper API: `include/ptxsim/testing/instruction_helpers.h`
  (`make_bra`, `make_bra_pred`, `make_nop`, `make_ret`)
- Helper API: `include/ptxsim/testing/predicates.h` (`setup_pred`)
