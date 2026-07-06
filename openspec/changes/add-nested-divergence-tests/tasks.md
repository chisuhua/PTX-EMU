## 1. Artifacts Commit (Phase 0 — per Checklist E, mandatory)

- [ ] 1.1 git add `openspec/changes/add-nested-divergence-tests/` (4 artifacts: proposal.md + design.md + specs/nested-divergence-coverage/spec.md + tasks.md)
- [ ] 1.2 git commit with message starting `docs(openspec): add add-nested-divergence-tests artifacts` and refs to debt audit §A-10

## 2. Single-Phase Implementation (~5h, Tier 1)

### 2.1 Empirical verification of helpers (pre-impl checklist)

- [ ] 2.1.1 Verify `make_bra(target)` signature and output (already in `tests/integration/divergence/test_post_barrier_reconvergence_simplegemm.cpp:48`)
- [ ] 2.1.2 Verify `make_bra_pred(target, pred, neg, reconv_pc)` output matches the audit-friendly text format (no "need-to-verify" gaps)
- [ ] 2.1.3 Verify `setup_pred(warp, mask)` writes per-lane predicate registers (`p1`, `p2`) — read source to confirm bit-to-lane mapping

### 2.2 Build the test (single file change)

- [ ] 2.2.1 Open `tests/integration/divergence/test_nested_divergence.cpp`
- [ ] 2.2.2 Append a new TEST_CASE block (after the existing
  `test_nested_predication`) implementing the design.md §1 PTX scenario
- [ ] 2.2.3 Use only existing helpers: `make_bra_pred`, `make_bra`,
  `make_nop`, `make_ret`, `step_warp`, `setup_pred`, `setup_block`
- [ ] 2.2.4 Verify per-lane register values per the design.md §2
  path table (lanes 0..7 / 8..15 / 16..31 each verify their own row)
- [ ] 2.2.5 Add a max-cycle guard so a regression deadlock fails the
  test instead of hanging forever (matches the existing
  `test_atom_exch.cpp` pattern: 5000-cycle cap)
- [ ] 2.2.6 Update the file's top header comment to note that
  `@%p bra` coverage now exists alongside the `setp+selp` coverage

### 2.3 Must NOT

- [ ] 2.3.1 Do NOT modify `WarpContext::handle_branch`,
  `simt_stack.cpp`, or any production code
- [ ] 2.3.2 Do NOT remove or modify the existing
  `test_nested_predication` block
- [ ] 2.3.3 Do NOT introduce `force_set_pc`, direct
  `set_active_mask` writes, or `qualifiers.back()` calls (per
  ptx-lessons-learned §1 §2 §5 + AGENTS.md §ANTI-PATTERNS)
- [ ] 2.3.4 Do NOT skip the max-cycle guard; a regression that
  hangs the CI is worse than one that fails the test

### 2.4 Verification Gates

- [ ] 2.4.1 **G1 (build)**: `cmake --build build -j$(nproc)` — 0 errors
- [ ] 2.4.2 **G2 (ctest divergence)**: `cd build && ctest -R
  integration.*divergence --output-on-failure` — 0 failures,
  including the new TEST_CASE
- [ ] 2.4.3 **G3 (no regression)**: `cd build && ctest -L unit
  --output-on-failure` — all unit tests still PASS
- [ ] 2.4.4 **G4 (PTX syntax)**: `bash tests/ptx/test_all_ptx.sh` —
  34/34 PASS (informational; new test is C++)
- [ ] 2.4.5 **G5 (sanity)**: `bash scripts/sanity.sh --quick` — all pass

## 3. Commit (single atomic per ptx-lessons-learned §3)

- [ ] 3.1 git stage: only `tests/integration/divergence/test_nested_divergence.cpp` (no other files unless hooks add generated content)
- [ ] 3.2 git commit with message:
  ```
  test(divergence): add two-level @%p bra nested divergence test (Fix #1)

  Closes A-10 (tests/integration/divergence/test_nested_divergence.cpp:106
  carried a TODO since 2026-05-08; the file's previous setp+selp scenario
  does NOT exercise the SIMT stack twice).

  Test:
  - 32-lane warp
  - Outer: @p1 bra L_then (lanes 0..15 take then, 16..31 fall through)
  - Inner: @p2 bra L_INNER (lanes 0..7 take inner-then, 8..15 inner-else)
  - Verify per-lane r1/r2/r3 match the path table
  - Max-cycle guard (5000) to fail fast on regression deadlock

  Refs: openspec/changes/add-nested-divergence-tests/specs/nested-divergence-coverage/spec.md
        openspec/changes/add-nested-divergence-tests/design.md §1 + §2
        docs/audits/debt-audit-2026-07-02.md §A-10
  ```

## 4. Revert Strategy

- [ ] 4.1 If G1-G5 fails: `git revert HEAD` restores the prior
  state (test-only commit, no production code to recover).
- [ ] 4.2 Investigate root cause; re-author in a new commit; do
  NOT amend.

## 5. Post-Phase Archive

- [ ] 5.1 `openspec validate --changes "add-nested-divergence-tests"` — must be valid
- [ ] 5.2 `openspec archive add-nested-divergence-tests --yes` — moves to `archive/2026-07-06-add-nested-divergence-tests/` and publishes spec to `openspec/specs/nested-divergence-coverage/spec.md`
- [ ] 5.3 git commit archive (Checklist G) with message:
  ```
  chore(openspec): archive add-nested-divergence-tests (Checklist G)

  A-10 RESOLVED.
  ...
  ```
- [ ] 5.4 Update `docs/roadmap/post-phase3-debt-roadmap.md §1.1 A-10` row to ✅
- [ ] 5.5 (Optional per openspec-archive-change skill) Append postmortem to `ptx-lessons-learned.md` if any new failure mode was encountered.

## Total time budget

- Phase 1: ~5h (per roadmap; conservative estimate for first cross-`make_bra_pred` + `step_warp` interaction in this file)
- Archive + roadmap sync: ~5min
