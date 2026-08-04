## Context

Phase 1+2 of `2026-07-27-god-class-refactor-sm-context` are applied: commits `59df7abf` (dedup reconvergence orchestration) and `01389c18` (ADR-0020 cpptlm injection). `sm_context.cpp` is now **862 lines** (verified `wc -l` on 2026-08-03, HEAD `041ae7d1`), down from 965. The remaining responsibility surface is still mixed:

| Responsibility | Current members | Lines (verified) |
|---|---|---|
| CTA admission / queue | `add_block` 130-204, `try_admit_pending_blocks` 206-258, `cleanup_finished_blocks` 628-643, `free_shared_memory` 645-665, `reserve_resources` 667-689, `release_resources` 691-695 | ~193 |
| Warp lifecycle | `update_state` 586-626, `select_next_group` 831-855, `suspend_and_switch` 856-862, `get_active_warps_count` 562-570, `get_active_threads_count` 572-580 | ~99 |
| SM barrier glue | thin glue inside `exe_once` and a few helpers; `BarrierModule` already owns the semantics | ~30 |
| `exe_once()` monolithic main loop | 333-558 | 226 |
| Other (init/destruct/print/utilities/heuristics) | various | ~314 |

`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2` sets the helper-component cap at ≤4 (existing: `sm_context_reconvergence.{h,cpp}`, `sm_context_cpptlm_inject.{h,cpp}`). This change extracts three cohesive helpers; total helpers become 5. The cap is bumped to ≤5 and the roadmap note is updated accordingly.

`src/ptxsim/core/AGENTS.md:89` and commit `d9172014` message both flag the 226-line `exe_once()` loop as "highly cohesive and not safely extractable in a single session". Decomposing it is deferred to a follow-up change. Reaching the original `<250` line target is therefore a **multi-change end-state**, not a single-change acceptance gate. The target for this change is `sm_context.cpp ≤ 600 lines` (net reduction ≥ 250 lines).

Verified invariant locations:

- `w->update_active_mask()` at `src/ptxsim/core/sm_context.cpp:362`.
- BUG-001 active_count synchronization comment block `src/ptxsim/core/sm_context.cpp:354-359`, with the key sentence "only updated by update_active_mask(). Without this fix, active_count…" at line 357.
- `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` 4-branch test for the byte-identical fallback (lessons-learned §14).
- `WarpContext` public API frozen by archive `2026-07-27-refactor-warp-context`.

## Goals / Non-Goals

**Goals**:

- `sm_context.cpp ≤ 600 lines` (`wc -l`).
- Three cohesive helper components, each with direct unit tests:
  - `sm_block_dispatch.{h,cpp}` (`sm_block_dispatch::`)
  - `sm_warp_lifecycle.{h,cpp}` (`sm_warp_lifecycle::`)
  - `sm_barrier_wrapper.{h,cpp}` (`sm_barrier_wrapper::`)
- Preserve `update_active_mask()` site + BUG-001 comment block (lessons-learned §1).
- Preserve `step_b` byte-identical fallback 4-branch test (lessons-learned §14).
- Zero regression on `tests/integration/barrier/*`, `tests/integration/divergence/*`, and the existing `tests/unit/sm/*` suite.
- Helper-component cap revised from ≤4 to ≤5, recorded in the roadmap.

**Non-Goals**:

- Modify WarpContext public API (C-18 freeze).
- Modify `SMContext::exe_once()` signature.
- Modify BarrierModule internal implementation.
- Decompose the 226-line `exe_once()` body (deferred to `exe-once-decomposition`).
- Touch CTAContext or ThreadContext interfaces.
- Introduce a new `Wbar` struct.

## Decisions

### Decision 1: Helper-namespace pattern, not new classes

**Choice**: Use the same anonymous-namespace helper pattern used by `sm_context_reconvergence.{h,cpp}` (`sm_reconvergence::`) and `sm_context_cpptlm_inject.{h,cpp}` (`sm_cpptlm_inject::`). Each new helper is a free-function namespace declared in `.h`, implemented in `.cpp`. `SMContext` friend-declares each namespace for direct private access.

**Rationale**: Mirrors the existing Phase 1+2 pattern (commits `59df7abf`, `01389c18`), minimises new public API surface, and lets the parent class stay the single integration point.

**Alternatives**:

- A. New `SmBlockDispatch` class instance — adds per-instance state and lifecycle; not needed because responsibilities are pure functions of `SMContext`.
- B. New `SmBlockDispatch` static class — closer, but the existing helper-namespace pattern is already adopted.
- C. **Adopted**: helper namespace + friend declaration.

### Decision 2: Phase ordering (block → lifecycle → barrier)

**Choice**: Phase 3 = block dispatch (largest extractor, ~193 lines). Phase 4 = warp lifecycle (~99 lines). Phase 5 = SM barrier wrapper (~30 lines, with go/no-go check). Each Phase is a revert-safe commit.

**Rationale**: Block dispatch is the largest coherent extractor and has minimal coupling to the remaining `exe_once()` body. Warp lifecycle follows because it consumes `WarpContext` (C-18 frozen) and is independent of block dispatch. Barrier wrapper is the smallest and most isolated; doing it last limits risk if its scope is later judged too thin to justify a component.

**Alternatives**:

- A. Lifecycle first — would couple earlier with WarpContext friend surface.
- B. Barrier first — smallest, but then phases 3 and 4 must follow with a more aggressive budget that could mask barrier glue drift.
- C. **Adopted**: block → lifecycle → barrier.

### Decision 3: Helper cap bumped to ≤5 (not ≤4)

**Choice**: Helper cap becomes ≤5 instead of ≤4. Recorded in `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2` as part of this change.

**Rationale**: The C-2 cap was set assuming 2 existing helpers + 1 new helper for CTA dispatch + 1 for barrier wrapper. The actual current state has 2 existing helpers; this change adds 3. Bumping the cap to ≤5 keeps the cap a real engineering constraint (not a tautology) while accommodating the proven extraction set.

**Alternatives**:

- A. Force ≤4 by merging `sm_warp_lifecycle` into `sm_block_dispatch` as a single `sm_block_warp_lifecycle.{h,cpp}` — would produce a file that still mixes CTA admission with warp lifecycle, violating the original "single responsibility per component" goal.
- B. **Adopted**: bump cap to ≤5.

### Decision 4: `update_active_mask()` ownership follows the scheduling site

**Choice**: The `w->update_active_mask()` call at `src/ptxsim/core/sm_context.cpp:362` and the BUG-001 comment block at lines 354-359 move with whichever helper ends up owning the call site. If both block dispatch and warp lifecycle want it, the call is duplicated explicitly with the comment preserved at both sites; the comment must remain byte-identical.

**Rationale**: Lessons-learned §1 explicitly requires line-level diff for cross-module state translations. Moving the comment with the call prevents the silent drift seen in `2026-07-03-dead-code-cleanup` (lessons-learned §6 case study).

**Alternatives**:

- A. Keep the call in `sm_context.cpp` after extraction — defeats the purpose of cohesive components.
- B. **Adopted**: move with the call site; comment preserved verbatim.

### Decision 5: Recursive lock audit before any helper extraction

**Choice**: Before each Phase commit, run a recursive-lock audit (grep `lock_guard` / `unique_lock` in the helpers' surrounding members and confirm no helper takes the same mutex that the public caller already holds). Document the audit result in the Phase commit message.

**Rationale**: Lessons-learned §2 documents a real recursive-lock deadlock in `BarrierModule::arrive()`.

**Alternatives**:

- A. Skip the audit — risks deadlocks that only surface in concurrency tests.
- B. **Adopted**: per-Phase audit before commit.

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|---|---|---|
| `update_active_mask()` site drift | active_count sync regression | §1 line-level diff + comment verbatim preservation per Decision 4 |
| Helper-namespace friend declaration exceeds least-privilege | private field exposure widens | Each Phase commit reviewed against `WarpContext` / `SMContext` friend list; helpers take only what they need |
| Recursive lock on shared mutex | deadlock under concurrency | §2 audit before each Phase commit |
| `step_b` byte-identical fallback regression | latent execution drift | keep 4-branch test green; verify `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` passes per Phase |
| Phase commits break integration tests independently | multi-day debug | Each Phase runs full `ctest --output-on-failure`; revert-on-regression rule |
| Bumping helper cap to 5 invites future debt | roadmap drift | cap documented in `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2` |
| `sm_barrier_wrapper` may prove too thin | over-engineering | If after Phase 4 the new file is < 50 lines, fold it back into `sm_block_dispatch.{h,cpp}` in a follow-up patch; this change's Phase 5 includes an explicit go/no-go check |

## Migration Plan

**Phase 0: Pre-work (30 min)**

1. Create baseline worktree at `.worktrees/baseline-god-class-p3` from current HEAD; build and run full ctest (per lessons-learned Checklist B).
2. `git worktree add .worktrees/refactor-god-class-p3 -b openspec/god-class-refactor-sm-context-phase3 HEAD`.
3. Record `wc -l src/ptxsim/core/sm_context.cpp = 862`.
4. Verify `src/ptxsim/core/sm_context.cpp:362 w->update_active_mask()` exists and comment block `:354-359` is intact.
5. Verify `ls openspec/changes/archive/*-refactor-warp-context` exists.
6. Verify `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` is present and currently passing.

**Phase 3: CTA block dispatch (3 h)**

1. Create `tests/unit/sm/test_sm_block_dispatch.cpp` (RED): three failing tests for admit happy path, overflow → pending queue, cleanup releasing resources.
2. Create `src/ptxsim/core/sm_block_dispatch.{h,cpp}` skeleton.
3. Move `add_block`, `try_admit_pending_blocks`, `cleanup_finished_blocks`, `free_shared_memory`, `reserve_resources`, `release_resources` into the namespace as free functions over `SMContext&`.
4. Friend-declare `sm_block_dispatch::` inside `SMContext`.
5. Update `src/ptxsim/core/CMakeLists.txt` to add the new sources.
6. Compile, run new tests (GREEN).
7. Recursive-lock audit (Decision 5).
8. Run `ctest --output-on-failure` — must stay green.
9. Verify `wc -l src/ptxsim/core/sm_context.cpp` ≤ 712 (≥150 net reduction).
10. Commit: `refactor(sm): extract CTA block dispatch to sm_block_dispatch.{h,cpp}`.

**Phase 4: warp lifecycle (2.5 h)**

1. Create `tests/unit/sm/test_sm_warp_lifecycle.cpp` (RED): three failing tests for warp registration, retirement via `update_state`, and `get_active_warps_count` consistency after retirement.
2. Create `src/ptxsim/core/sm_warp_lifecycle.{h,cpp}` skeleton.
3. Move `update_state`, `select_next_group`, `suspend_and_switch`, `get_active_warps_count`, `get_active_threads_count`.
4. If `w->update_active_mask()` ownership follows any of these members to the new helper, move the comment block too (Decision 4).
5. Friend-declare `sm_warp_lifecycle::` inside `SMContext`.
6. Update `src/ptxsim/core/CMakeLists.txt`.
7. Compile, run new tests (GREEN).
8. Recursive-lock audit (Decision 5).
9. Run `ctest --output-on-failure` — must stay green.
10. Verify `wc -l src/ptxsim/core/sm_context.cpp` ≤ 600.
11. Commit: `refactor(sm): extract warp lifecycle to sm_warp_lifecycle.{h,cpp}`.

**Phase 5: SM barrier wrapper (1.5 h, with go/no-go check)**

1. Decide go/no-go for the wrapper based on remaining barrier glue in `sm_context.cpp`. If `< 50` lines remain after Phase 4, fold this Phase into a small follow-up patch instead of a new file.
2. If go: create `tests/unit/sm/test_sm_barrier_wrapper.cpp` (RED): two failing tests for `cta_context->get_barrier_module()` delegation and null-barrier fallback.
3. Create `src/ptxsim/core/sm_barrier_wrapper.{h,cpp}` and move SM-level barrier glue.
4. Friend-declare `sm_barrier_wrapper::` inside `SMContext`.
5. Update `src/ptxsim/core/CMakeLists.txt`.
6. Compile, run new tests (GREEN).
7. Recursive-lock audit.
8. `ctest --output-on-failure` green.
9. Commit: `refactor(sm): extract SM barrier wrapper to sm_barrier_wrapper.{h,cpp}`.

**Phase 6: final validation (1 h)**

1. `wc -l src/ptxsim/core/sm_context.cpp ≤ 600`.
2. `src/ptxsim/core/sm_context.cpp` still references the same WarpContext public API surface (no new calls, no removed calls).
3. `src/ptxsim/core/sm_context.h` `exe_once()` signature unchanged.
4. `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` 4-branch test passes.
5. `tests/unit/sm/test_sm_block_dispatch.cpp`, `test_sm_warp_lifecycle.cpp`, `test_sm_barrier_wrapper.cpp` (if created) all pass.
6. `tests/integration/barrier/*`, `tests/integration/divergence/*` pass.
7. `grep -n 'update_active_mask' src/ptxsim/core/sm_*.cpp` accounts for every call site (no orphan or duplicate).
8. `grep -n 'BUG-001' src/ptxsim/core/sm_*.cpp` accounts for the comment block (no orphan).
9. `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2` updated: cap bumped to ≤5, status note pointing to follow-up `exe-once-decomposition`.
10. `src/ptxsim/core/AGENTS.md` table updated to include the three new helper components.

**Rollback strategy**

- Each Phase commit is independently revertable.
- `git revert HEAD~N..HEAD` for any Phase that introduces regressions; the remaining Phases continue without that Phase's changes.

## Open Questions

- None at design close. `exe_once()` decomposition scope and method remain the explicit follow-up change (`exe-once-decomposition`).