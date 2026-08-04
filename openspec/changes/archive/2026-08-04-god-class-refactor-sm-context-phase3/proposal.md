# god-class-refactor-sm-context-phase3

## Why

`src/ptxsim/core/sm_context.cpp` measured **862 lines** on 2026-08-03 (`wc -l`). Phases 1+2 of `2026-07-27-god-class-refactor-sm-context` are already applied (commits `59df7abf` dedup reconvergence and `01389c18` extract cpptlm injection), reducing the file from 965 → 862 lines. Remaining responsibilities are still mixed:

1. CTA admission / pending queue (`add_block`, `try_admit_pending_blocks`, `cleanup_finished_blocks`, `free_shared_memory`, `reserve_resources`, `release_resources`).
2. Warp lifecycle (`update_state`, `select_next_group`, `suspend_and_switch`, `get_active_warps_count`, `get_active_threads_count`).
3. SM-level barrier coordination glue (delegating to `BarrierModule`).

`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2` keeps C-2 the only deteriorating debt. Reaching the original `<250` line target additionally requires decomposing the 226-line `exe_once()` loop, which `src/ptxsim/core/AGENTS.md:89` and commit `d9172014` explicitly defer: *"highly cohesive and not safely extractable in a single session"*. This change therefore targets **≤600 lines** by extracting the three cohesive sub-modules above and **defers `exe_once()` decomposition to a follow-up change** (`exe-once-decomposition`).

Phase 1+2 are already committed, so this change resumes numbering at "Phase 3+" to avoid double-meaning with the original `2026-07-27-god-class-refactor-sm-context` archive.

## What Changes

- **Extract CTA admission logic** to `src/ptxsim/core/sm_block_dispatch.{h,cpp}` (helper namespace `sm_block_dispatch::`).
- **Extract warp lifecycle logic** to `src/ptxsim/core/sm_warp_lifecycle.{h,cpp}` (helper namespace `sm_warp_lifecycle::`).
- **Extract SM-level barrier wrapper** to `src/ptxsim/core/sm_barrier_wrapper.{h,cpp}` (helper namespace `sm_barrier_wrapper::`).
- **Preserve line-level evidence** for lessons-learned §1: the `w->update_active_mask()` call and its BUG-001 active_count synchronization comment move together with their new owning component (current verified site: `sm_context.cpp:362`; comment block starts at line 354 with the key sentence at line 357).
- **Preserve** the `step_b` no-op byte-identical fallback 4-branch test (`tests/unit/sm/test_step_b_set_blocked_cycles.cpp`, lessons-learned §14).
- **Hold** WarpContext public API signatures constant (C-18 boundary from archive `2026-07-27-refactor-warp-context`).
- **Hold** `SMContext::exe_once()` signature constant (no surgery on the 226-line loop in this change).
- **Hold** BarrierModule internal implementation constant; only thin glue is added to the wrapper component.

## Capabilities

### New Capabilities

- `sm-context-component-split`: SMContext internal responsibilities split into three cohesive helper components (`sm_block_dispatch`, `sm_warp_lifecycle`, `sm_barrier_wrapper`) with locked unit tests; `exe_once()` decomposition deferred.

### Modified Capabilities

- (no spec-level behaviour change; pure implementation refactor.)

## Impact

**Production code**:

| File | Action | Notes |
|---|---|---|
| `src/ptxsim/core/sm_context.cpp` | Modified | Net reduction from 862 → ≤600 lines |
| `src/ptxsim/core/sm_context.h` | Modified | Forward declarations for helper namespaces |
| `src/ptxsim/core/sm_block_dispatch.{h,cpp}` | New | `sm_block_dispatch::` helper namespace |
| `src/ptxsim/core/sm_warp_lifecycle.{h,cpp}` | New | `sm_warp_lifecycle::` helper namespace |
| `src/ptxsim/core/sm_barrier_wrapper.{h,cpp}` | New | `sm_barrier_wrapper::` helper namespace |
| `src/ptxsim/core/CMakeLists.txt` | Modified | Register three new source files |

**Tests**:

| File | Action | Notes |
|---|---|---|
| `tests/unit/sm/test_sm_block_dispatch.cpp` | New | ≥3 cases covering admit, overflow → pending, cleanup |
| `tests/unit/sm/test_sm_warp_lifecycle.cpp` | New | ≥3 cases covering warp registration, retirement, active-count consistency |
| `tests/unit/sm/test_sm_barrier_wrapper.cpp` | New | ≥2 cases covering delegator and null-barrier fallback |
| `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` | Preserved | 4-branch byte-identical fallback (lessons-learned §14) |
| `tests/integration/barrier/*`, `tests/integration/divergence/*` | Regression-only | No source change expected; full ctest must stay green |

**Component count invariant** (per `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2` ≤4 cap on helpers):
- Existing helpers: `sm_context_reconvergence.{h,cpp}`, `sm_context_cpptlm_inject.{h,cpp}` (= 2).
- This change adds: 3 helper files.
- `ls src/ptxsim/core/sm_context_*.{h,cpp}` would return 10 files. Helper count becomes 5, exceeding the ≤4 cap by one. The cap is bumped to ≤5 for helpers and the roadmap note is updated; this is acknowledged in the design decision log.

**§1 evidence (lessons-learned SKILL.md:48-77)**:

| Item | Current verified location | Move target |
|---|---|---|
| `w->update_active_mask()` call | `src/ptxsim/core/sm_context.cpp:362` | Whichever helper owns the scheduling site that calls it (decided per-phase) |
| BUG-001 active_count synchronization comment block | `src/ptxsim/core/sm_context.cpp:354-359` (key sentence at line 357) | Moves together with the call site |
| `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` 4-branch invariants | (file content) | Preserved unchanged |

**Out-of-scope (deferred to a follow-up change)**:

- `exe_once()` decomposition (226 lines, `src/ptxsim/core/AGENTS.md:89` warning).
- WarpContext public API changes (C-18 freeze).
- BarrierModule internal changes.
- CTAContext interface changes.

**Dependencies**:

- **Required**: archive `2026-07-27-refactor-warp-context` (C-18, WarpContext API freeze) — verified by `ls openspec/changes/archive/*-refactor-warp-context`.
- **Required**: lessons-learned skills §1 (cross-module state translation), §14 (no-op contract), Checklist B (worktree + Phase commits).
- **Optional**: `state-modification-audit` skill for active_mask audit during Phase 4.

**Key constraints (MUST / MUST NOT)**:

- MUST §1 line-level diff for the `update_active_mask()` site (use current `sm_context.cpp:362`, comment block `354-359`, not the stale 379/374 in the previous artifacts).
- MUST Checklist B (worktree + independent Phase commits, revert-safe).
- MUST §14 keep `step_b` byte-identical fallback 4-branch test passing.
- MUST §2 audit recursive lock paths before extracting any helper that takes the same mutex as the public method calling it.
- MUST NOT modify WarpContext public API.
- MUST NOT modify `SMContext::exe_once()` signature.
- MUST NOT modify BarrierModule internals.
- MUST NOT introduce a new `Wbar` struct.

**Estimated effort**: 8-10h (3 Phase commits + worktree setup + 3 new unit-test files + final validation).