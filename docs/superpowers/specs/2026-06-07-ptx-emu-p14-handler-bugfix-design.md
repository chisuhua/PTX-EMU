# P1-4 Handler Bug Fixes — Design

**Date:** 2026-06-07
**Status:** Draft (pending user review)
**Parent:** [`2026-06-06-ptx-emu-test-coverage-roadmap.md`](./2026-06-06-ptx-emu-test-coverage-roadmap.md) §2 + [`docs/developer-guide/KNOWN_ISSUES.md` §P1-4.1, §P1-4.2](../../developer-guide/KNOWN_ISSUES.md)
**Estimated effort:** 1-2 hours
**Out of scope:** New PTX instructions, handler refactoring

---

## 1. Goal

Fix the 2 latent handler/test bugs surfaced by P1-4 work and re-enable the 5 SKIP'd TEST_CASEs. After completion:
- All 21 Tier 3 integration tests pass actively (0 SKIP)
- `KNOWN_ISSUES.md §P1-4.1 and §P1-4.2` entries are removed
- `sanity.sh --tier 3` continues to exit 0

## 2. Refined root cause analysis (2026-06-07)

Prior investigation revealed the actual root causes differ from the original KNOWN_ISSUES.md guesses:

### §P1-4.1: CvtHandler missing f32→s32 case (REAL HANDLER BUG)

**Symptom:** `cvt.s32.f32 r2, r1` writes 0 to r2 (when r1 has non-zero float bits).

**Suspected location:** `src/ptxsim/instructions/arithmetic_conversion.cpp:886-962` — the 4-byte `case` with `dst_is_int && src_is_float` branch.

**Why we believe this is real:** The cvt test (line 145-180) does **not** include `make_mov` — r1 is directly seeded with float bits. So if the test fails, it's a handler bug, not a test setup bug. The f64→s64 cvt test (which was SKIP'd) actually passes after SKIP removal — so that case works. The f32→s32 case is the one that genuinely fails.

**Hypothesis:** The handler at lines 886-962 has the code path but may:
- Use wrong default branch (line 959 `*(uint32_t *)dst = static_cast<uint32_t>(temp);` writes 0 in some edge case)
- Have a missing `else` clause that causes `dst` to never be written for some sub-case
- The `switch(dst_bytes)` may not have an `else` after the float-dst branch (line 862-885), causing fallthrough into the int-dst branch (line 886) even when `dst_is_float=true` — but only for some combinations

**Investigation needed:** Read the full switch statement + check the working f64→s64 case vs failing f32→s32 case.

### §P1-4.2: NOT a handler bug — TEST BUG (test_float_arith.cpp make_mov)

**Refuted:** Prior assumption was that `AddHandler`/`MulHandler`/`FmaHandler` don't branch on `Q_F32`. **Investigation 2026-06-07 disproves this.**

**Actual root cause:** `tests/integration/ptx/test_float_arith.cpp` has `stmts.push_back(make_mov("r1", "tid.x"));` at PC=0 in all 4 TEST_CASEs. The `make_mov` overwrites the seeded r1 (which held float bits) with `tid.x` (integer lane_id). Then `fadd r2, r1, r1` does float add of integer lane_id bits:
- lane 0: 0 + 0 = 0.0f → 0x00000000. Coincidentally matches `expected = bits(0.0f) = 0`. ✓
- lane 1: bits(1) = 0x00000001 = 1.4e-45 (denormal). Float add: 1.4e-45 + 1.4e-45 = 2.8e-45 = bits(0x00000002). But test expects `bits(2.0f) = 0x40000000`. ✗
- The fsub test passes (any value - same value = 0).

**Fix:** Remove the `stmts.push_back(make_mov("r1", "tid.x"));` line from the 3 SKIP'd TEST_CASEs (fadd, fmul, ffma). The fsub test also has it but it passes by coincidence — leave it or remove it for consistency.

### Why this matters

The original KNOWN_ISSUES.md §P1-4.2 speculated about handler bugs. That was wrong. The real bug is in the test code that was committed in P1-4. Fixing the test code is trivial (1 line deletion per TEST_CASE). The handler investigation for §P1-4.1 is the real work.

## 3. File list

### 3.1 Modified test files (1)
- `tests/integration/ptx/test_float_arith.cpp` — remove `make_mov("r1", "tid.x")` line from 4 TEST_CASEs (or 3 SKIP'd + 1 fsub)
- Restore the test bodies (currently have `SKIP(...)` at top — remove SKIPs)

### 3.2 Modified handler file (1)
- `src/ptxsim/instructions/arithmetic_conversion.cpp` — investigate and fix the 4-byte `case` in `CvtHandler::processOperation` to make `cvt.s32.f32` work

### 3.3 Modified docs (1)
- `docs/developer-guide/KNOWN_ISSUES.md` — remove §P1-4.1 and §P1-4.2 entries

### 3.4 Untouched
- All other handler files (AddHandler/MulHandler/FmaHandler WORK correctly per §P1-4.2 refutation)
- All other test files
- Sanity scripts

## 4. Architecture / data flow

### 4.1 Test fix (§P1-4.2)

Remove the `make_mov` line. The test becomes:
```cpp
// Before (current SKIP'd state):
stmts.push_back(make_mov("r1", "tid.x"));  // PC=0 — overwrites seeded r1
stmts.push_back(make_fadd("r2", "r1", "r1")); // PC=1
stmts.push_back(make_ret());

// After fix:
stmts.push_back(make_fadd("r2", "r1", "r1")); // PC=0
stmts.push_back(make_ret());
```

Now r1 retains the float bits seeded by `set_reg_per_lane_u32`, and the fadd at PC=0 does proper float add.

### 4.2 Handler fix (§P1-4.1)

Read `arithmetic_conversion.cpp:861-980` to identify the exact issue. Likely candidates:
- Line 886-885 boundary: `if (dst_is_float) { ... } else { ... }` — verify no fallthrough or missing `break`
- Line 916-961: rounding mode switches — verify the default case (line 959 `*(uint32_t *)dst = static_cast<uint32_t>(temp);`) is correct
- The `dst` pointer passed to processOperation may not be the same as the one written to (but DBG output showed `acquire_register` returning valid pointer for r2)

Investigation step before fix: add a debug print to `CvtHandler::processOperation` to confirm the function is entered for f32→s32 and what `dst`/`src`/`dst_qualifiers`/`src_qualifiers` look like.

## 5. Success criteria

- [ ] `cvt.f32.s32` test passes (no SKIP)
- [ ] `cvt.f64.s64` test passes (no SKIP, already passes after SKIP removal)
- [ ] `fadd.f32` test passes (no SKIP, after removing `make_mov` from test)
- [ ] `fmul.f32` test passes (no SKIP, after removing `make_mov`)
- [ ] `ffma.f32` test passes (no SKIP, after removing `make_mov`)
- [ ] `fsub.f32` test still passes (already does, coincidentally — may keep or remove `make_mov` for consistency)
- [ ] `ctest -L "integration;ptx"` shows 21/21 PASS, 0 SKIP
- [ ] `KNOWN_ISSUES.md §P1-4.1` and §P1-4.2 removed
- [ ] `sanity.sh --tier 3` still exits 0
- [ ] No regression in other tests

## 6. Risks

| Risk | Mitigation |
|---|---|
| CvtHandler f32→s32 fix may break other cvt paths (e.g., f64→s64) | Read the full switch before fix; test all 4 cvt TEST_CASEs after fix |
| Removing `make_mov` from float tests may surface new handler bugs | Run all 4 float tests after fix; if new bugs appear, document in NEW KNOWN_ISSUES section |
| Test fix may not be the actual root cause for §P1-4.2 (deeper bug) | If test fix doesn't make tests pass, re-investigate with debug prints |

## 7. Investigation order (recommended)

1. **First**: Remove `make_mov` from 3 SKIP'd float tests (test fix). Remove SKIPs. Run tests.
2. **If float tests now pass**: §P1-4.2 was a test bug. Done. Move to §P1-4.1.
3. **If float tests still fail**: Real handler bug exists. Add debug print to AddHandler to see what's happening.
4. **For §P1-4.1**: Add debug print to CvtHandler to see if it's called for f32→s32, and what `dst_qualifiers`/`src_qualifiers` look like.
5. **After fixes verified**: Remove §P1-4.1 and §P1-4.2 from KNOWN_ISSUES.md.
6. **Final commit**: Combine all fixes (test + handler + docs) into one logical commit.

## 8. Out of scope (intentional)

- New PTX instructions
- Handler refactoring
- WMMA / MMA
- P2 (enable commented tests) — separate roadmap item
- Re-running P1-3 / P1-4 plans after fix (test files already written; just removing SKIPs)
