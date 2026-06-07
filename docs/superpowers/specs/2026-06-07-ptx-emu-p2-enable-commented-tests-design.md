# P2: Re-enable Commented-Out Tests — Design

**Date:** 2026-06-07
**Status:** Draft (pending user review)
**Parent:** [`2026-06-06-ptx-emu-test-coverage-roadmap.md`](./2026-06-06-ptx-emu-test-coverage-roadmap.md) §4
**Estimated effort:** 3-5 hours total
**Out of scope:** `test_wmma` (explicitly SKIP per roadmap — requires WMMA implementation)

---

## 1. Goal

Re-enable 3 of 4 commented-out tests documented in `KNOWN_ISSUES.md §D1.2`. After completion:
- `unit_barrier_verification` passes
- `test_cfg_debug` passes (or is removed if scope too large)
- `unit_cc_register` passes
- `KNOWN_ISSUES.md §D1.2` updated to mark P2-1/2/3 as ENABLED (P2-4 stays DISABLED)

---

## 2. Investigation findings (2026-06-07)

### P2-1: `unit_barrier_verification`

**Blocker:** `SIMTStackEntry` / `simt_stack` not in scope (per `tests/unit/CMakeLists.txt:46-50`)

**Verified root cause:** `tests/unit/barrier/test_barrier_verification.cpp` includes `ptxsim/wbar.h` and `ptxsim/thread_state.h` but does **NOT** include `ptxsim/simt_stack.h` where `SIMTStack` and `SIMTStackEntry` are defined (`include/ptxsim/simt_stack.h:12,23`).

**API match verified:** All methods used in the test exist in the current `SIMTStack`:
- `push(const SIMTStackEntry&)` ✓
- `pop()` ✓
- `top()` ✓
- `depth()` ✓
- `empty()` ✓
- `check_reconvergence(const std::array<ThreadState, 32>&)` ✓

**Fix scope:** Add 1 include line. ~5 minutes.

### P2-2: `test_cfg_debug`

**Blocker:** `PtxVisitor::getKernels` doesn't exist (per `tests/CMakeLists.txt:187-189`)

**Verified root cause:** The PtxVisitor API has been completely rewritten. Current API (`include/ptx_parser/ptx_visiter.h:13-126`):
- Constructor requires `PtxContext &context` parameter (no default constructor)
- No `getKernels()` method
- No `visit(string)` method (uses ANTLR4 `visitPtxFile(PtxFileContext*)` instead)
- The test (`tests/ptx/test_cfg_debug.cpp:24-27`) constructs `PtxVisitor visitor;` with no args and calls `visitor.getKernels()` — both fail to compile

**The test is a standalone program** (uses `int main`, not Catch2). The old `PtxVisitor` API was a one-off debug tool. Restoring it would require either:
- **Option A:** Rewrite the test to use the new PtxVisitor API + new PtxContext setup
- **Option B:** Remove the test entirely (it was a one-off debug utility, not a regression test)
- **Option C:** Convert to a Catch2 test that exercises the new API

**Fix scope:** Option B is cleanest. The test was a debug tool, not a regression check. Removing it loses nothing. Option A/C would require ~2 hours to wire up the new ANTLR4 pipeline.

**Recommendation:** Option B (remove). The `ptx/` directory already has `test_cfg_edge_cases.cpp` as the main CFG test.

### P2-3: `unit_cc_register`

**Blocker:** `subc_handler` undeclared + non-Catch2 format (per `tests/unit/CMakeLists.txt:201-208`)

**Verified root cause:** The test (`tests/unit/common/test_cc_register.cpp`) uses:
- `ADDC addc_handler;` and `SUBC subc_handler;` — **Both classes exist** at `src/ptxsim/instructions/arithmetic_ext.cpp:11,242` (verified in P1-4 work)
- `ConditionCodeRegister::set_cc_reg(CARRY_INDEX, true)` and `get_carry()` — **Need to verify these methods exist**
- `ThreadContext::get_condition_codes()` / `set_condition_codes()` — **Need to verify these methods exist**
- `process_operation(&context, operands, qualifiers)` — **Verify signature matches** (the test uses the raw signature; the modern API is `processOperation(ThreadContext*, void**, std::vector<Qualifier>&, const std::vector<char>*)`)

**Likely issues beyond "subc_handler undeclared":**
- Method name casing: test uses `process_operation` (snake_case), current API uses `processOperation` (camelCase)
- Qualifier type: test uses `std::vector<Qualifier>` (value), current API uses `const std::vector<Qualifier>&` (const ref)
- Test format: uses `void test_cc_register()` + `int main()` + `std::cout` + `assert()` — not Catch2 format

**Fix scope:** Rewrite to Catch2 format with corrected API calls. ~2-3 hours.

---

## 3. File list

### 3.1 P2-1 (1 file modified)
- `tests/unit/barrier/test_barrier_verification.cpp` — add `#include "ptxsim/simt_stack.h"`
- `tests/unit/CMakeLists.txt` — uncomment `unit_barrier_verification` entry

### 3.2 P2-2 (1 file removed OR 2 files rewritten)
- **Recommended:** Remove `tests/ptx/test_cfg_debug.cpp` + uncomment in `tests/CMakeLists.txt` (just delete the comment)
- **Alternative:** Rewrite test + CMakeLists — ~2 hours, low value (was a debug tool)

### 3.3 P2-3 (1 file rewritten, 1 file modified)
- `tests/unit/common/test_cc_register.cpp` — rewrite to Catch2 format with API corrections
- `tests/unit/CMakeLists.txt` — uncomment `unit_cc_register` entry

### 3.4 Docs
- `docs/developer-guide/KNOWN_ISSUES.md` — update §D1.2 to mark P2-1/2/3 as ENABLED

---

## 4. Architecture

### 4.1 P2-1 fix

```cpp
// Add to test_barrier_verification.cpp after line 3:
#include "ptxsim/simt_stack.h"
```

Then uncomment in `tests/unit/CMakeLists.txt:49-51`:
```cmake
add_catch_test(unit_barrier_verification
    barrier/test_barrier_verification.cpp
)
set_tests_properties(unit_barrier_verification PROPERTIES LABELS "unit;barrier;verification")
```

### 4.2 P2-2 fix (Option B: remove)

```bash
git rm tests/ptx/test_cfg_debug.cpp
# Remove the disabled comment block from tests/CMakeLists.txt
```

### 4.3 P2-3 fix

Rewrite `test_cc_register.cpp` from `void test_cc_register()` to:
```cpp
TEST_CASE("CC register: ADDC with carry flag", "[cc][addc]") { ... }
TEST_CASE("CC register: SUBC with borrow flag", "[cc][subc]") { ... }
TEST_CASE("CC register: no .cc means no update", "[cc][addc]") { ... }
```

Each uses `REQUIRE` instead of `assert`, `SECTION` for sub-cases, and the modern API.

---

## 5. Success criteria

- [ ] P2-1: `ctest -R "unit_barrier_verification"` passes
- [ ] P2-2: `tests/ptx/test_cfg_debug.cpp` removed (or rewritten if user prefers)
- [ ] P2-3: `ctest -R "unit_cc_register"` passes
- [ ] `KNOWN_ISSUES.md §D1.2` updated: P2-1 ✓, P2-2 ✓ (removed) or ⚠️, P2-3 ✓
- [ ] `sanity.sh --tier 6` exits 0 (barrier verification is in Tier 6)

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| P2-3 `ConditionCodeRegister` API doesn't match (methods renamed) | Investigate exact API before rewrite; adapt test |
| P2-2 removal is "deletion of a test" — may be seen as loss | The test was a one-off debug tool, not a regression check; CFG is covered by `test_cfg_edge_cases.cpp` |
| P2-3 rewrite introduces subtle assertion differences | Use original assertions verbatim; only change format (void→TEST_CASE, assert→REQUIRE) |

---

## 7. Out of scope (intentional)

- `test_wmma` (explicitly SKIP per roadmap)
- New tests beyond re-enabling existing ones
- Refactoring P2 tests to a new style
