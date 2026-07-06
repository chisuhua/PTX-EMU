# Design: add-nested-divergence-tests

> **Single-Phase change (test-only)**: closes A-10
> **No handler code changes**: this is a coverage addition

---

## 1. PTX Scenario (Conceptual)

The new TEST_CASE drives a 32-lane warp through the following
control-flow graph (PTX-equivalent; built via `make_bra_pred` /
`make_nop` / `make_ret` helpers):

```
                @p1 bra L_then          ; ── outer divergence ──
                mov.u32  r1, 1000        ; (lanes NOT taking L_then)
                bra      L_END          ; explicit jump over then-block
L_then:         mov.u32  r1, 100         ; ── (lanes taking outer-then) ──
                @p2 bra L_INNER         ; ── inner divergence ──
                mov.u32  r2, 300        ; (inner-fall-through arm)
                bra      L_END
L_INNER:        mov.u32  r2, 200        ; (inner-then arm)
L_END:          add.u32  r3, r1, r2    ; ── unified convergence ──
                ret
```

## 2. Per-lane Expectations

Given a setup similar to the existing
`test_divergence_sync_convergence.cpp` pattern:

| Lane group | `p1` | `p2` | `r1` | `r2` | `r3` (after `add`) |
|-----------|-----|-----|------|------|---------------------|
| 0..7 | true | true | 100 | 200 | 300 |
| 8..15 | true | false | 100 | 300 | 400 |
| 16..31 | false | n/a | 1000 | 0 (untouched) | 1000 |

Notes on the assertions:
- `r2` for lanes 16..31 is **never written** in the program; it stays
  at whatever initial value the register bank hands out. The test
  chooses zero-init for predictability.
- `r3` is `r1 + r2` for every lane at L_END; lanes 16..31 see
  `r1=1000 + r2=0 = 1000` (not `r2=300` or `r2=200`); this is the
  crucial invariant that proves the inner branch correctly skips
  non-applicable lanes.

## 3. SIMT Stack Depth Profile

The SIMT stack discipline is the actual property under test, even
though per-lane `r1/r2/r3` is what we observe.

| Event | SIMT Stack Depth |
|-------|-----------------|
| Enter test (warp created) | 0 |
| Outer `@p1 bra` evaluated, divergent masks split | 1 |
| Inner `@p2 bra` evaluated, inner masks split | 2 (peak) |
| Inner `bra L_END` reconverges | 1 |
| Outer fall-through `bra L_END` reconverges | 0 |

The peak depth of 2 is the property that the existing test file's
`setp+selp` variant cannot exercise (selp doesn't push the stack at
all). See AGENTS.md §DUAL STATE MECHANISM for the per-thread PC +
stack invariant.

## 4. Implementation Strategy

### 4.1 What gets added

One new TEST_CASE appended to
`tests/integration/divergence/test_nested_divergence.cpp`, **after**
the existing `test_nested_predication` block (no interference with
the existing `setp+selp` test which the audit said to keep). The new
case builds the program above via:

- `ptxsim::testing::make_bra_pred(target_label, pred_reg, false, reconv_pc)` for the two `@%p bra`
- `ptxsim::testing::make_bra(target_label)` for the unconditional
  `bra L_END` fall-through (lifts lanes 16..31 over the then-block)
- `ptxsim::testing::make_nop()` for padding (not strictly needed if we
  build a minimal PC count, but the existing tests pad for readability)
- `ptxsim::testing::make_ret()`
- `ptxsim::testing::step_warp(warp, stmts)` to drive each cycle
- `ptxsim::testing::setup_pred(warp, mask)` to assign the predicate
  register per lane (16-high for `p1`, 8-high for `p2` of the
  outer-then group)

### 4.2 What does NOT get added

- No new source file; one TEST_CASE in the existing file is enough.
- No header changes (helpers already in
  `include/ptxsim/testing/instruction_helpers.h` and
  `include/ptxsim/testing/predicates.h`).
- No CMakeLists.txt change (tests already in the divergence target).

## 5. Helpers That Will Be Used (no changes)

From `include/ptxsim/testing/instruction_helpers.h`:
- `make_bra(target)` — lines 587+
- `make_bra_pred(target, pred, neg, reconv_pc)` — lines 601+
- `make_nop()` — already used in `test_divergence_sync_convergence.cpp`
- `make_ret()` — already used

From `include/ptxsim/testing/predicates.h`:
- `setup_pred(warp, mask)` — assigns predicate register per lane

## 6. Test Assertions

| # | Assertion | Verification |
|---|----------|--------------|
| 1 | All 32 lanes reach unified convergence | loop `step_warp` until `ret_pc`; verify `is_finished` |
| 2 | SIMT stack peaks at 2 | (note: peek via debug trace, not strict assertion; CfgDebug may not be wired into the integration test path — verification-by-side-effect via per-lane register values is the primary signal) |
| 3 | Lanes 0..7: r1=100, r2=200, r3=300 | REQUIRE loop |
| 4 | Lanes 8..15: r1=100, r2=300, r3=400 | REQUIRE loop |
| 5 | Lanes 16..31: r1=1000, r3=1000 | REQUIRE loop |
| 6 | No deadlock / infinite loop | max-cycle guard (similar to `test_atom_exch`) |

## 7. Edge Cases

- If `p1` is set true for all 32 lanes: tests degenerate to a
  single-divergent-arm scenario (no outer fall-through). The test
  arms `p1=0xFFFF0000` (lanes 0..15 only), so this branch is not
  exercised; future-proofing deferred to a follow-up change.
- If `p2` is true for all lanes (in the outer-then group): the
  inner-else (`mov r2, 300`) is unreachable; this is a valid
  PTX pattern and the test would still verify all 16 outer-then lanes
  take the `L_INNER` branch. We can add a second variant if needed.

## 8. Verification Plan

```bash
# 1. Build target
cmake --build build -j$(nproc)

# 2. Run integration divergence tests (regression scope)
cd build && ctest -R "integration.*divergence" --output-on-failure

# 3. PTX syntax (informational — new test is C++ not PTX)
cd .. && bash tests/ptx/test_all_ptx.sh

# 4. Broad sanity
bash scripts/sanity.sh --quick
```

## 9. Revert Strategy

Single atomic commit. Revert via `git revert HEAD`; no cross-Phase
state to restore (this is a test-only addition; production code
unchanged).

## 10. Refs

- Debt audit: `docs/audits/debt-audit-2026-07-02.md` §2.1 A-10
- File with TODO: `tests/integration/divergence/test_nested_divergence.cpp:106`
- Reference: `tests/integration/divergence/test_post_barrier_reconvergence_simplegemm.cpp`
- Helper API: `include/ptxsim/testing/instruction_helpers.h` + `predicates.h`
