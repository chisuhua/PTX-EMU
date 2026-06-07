# P2: Re-enable Commented-Out Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-enable 3 of 4 commented-out tests per `KNOWN_ISSUES.md §D1.2` (P2-1, P2-2, P2-3). P2-4 (`test_wmma`) remains disabled per roadmap.

**Architecture:** Minimal per-test fixes. P2-1 is a 1-line include addition. P2-2 is a file removal. P2-3 is a format rewrite (void→TEST_CASE, assert→REQUIRE). No handler changes.

**Tech Stack:** C++20, Catch2 v3, PTX-EMU ptxsim.

**Parent spec:** [`docs/superpowers/specs/2026-06-07-ptx-emu-p2-enable-commented-tests-design.md`](../specs/2026-06-07-ptx-emu-p2-enable-commented-tests-design.md)

---

## CRITICAL PRE-IMPLEMENTATION NOTES

1. **P2-2 is a REMOVAL, not a fix.** `test_cfg_debug.cpp` uses a PtxVisitor API that no longer exists. The test was a one-off debug tool, not a regression check. Remove it entirely.
2. **P2-3 requires verification of `ConditionCodeRegister` API** before rewriting. The test uses `set_cc_reg`, `get_carry`, `get_condition_codes` — verify these exist in current `include/ptx_ir/ptx_types.h` or `ptxsim/thread_context.h`.
3. **Do NOT change `KNOWN_ISSUES.md §D1.2` P2-4 (test_wmma) — it stays disabled per roadmap.**

---

## Task 1: P2-1 — Re-enable `unit_barrier_verification`

**Files:**
- Modify: `tests/unit/barrier/test_barrier_verification.cpp` (add 1 include)
- Modify: `tests/unit/CMakeLists.txt` (uncomment entry)

- [ ] **Step 1: Add the missing include to test_barrier_verification.cpp**

Use Edit tool. Current header block (lines 1-6):
```cpp
#include "catch_amalgamated.hpp"
#include "ptxsim/wbar.h"
#include "ptxsim/thread_state.h"
#include <array>

using namespace ptxsim;
```

Add `#include "ptxsim/simt_stack.h"` after `#include "ptxsim/thread_state.h"`:
```cpp
#include "catch_amalgamated.hpp"
#include "ptxsim/wbar.h"
#include "ptxsim/thread_state.h"
#include "ptxsim/simt_stack.h"
#include <array>

using namespace ptxsim;
```

- [ ] **Step 2: Uncomment the `unit_barrier_verification` entry in CMakeLists.txt**

In `tests/unit/CMakeLists.txt` around lines 46-51, find:
```cmake
# test_barrier_verification.cpp — Catch2 v1 header swapped to project amalgamated
# (D1.2 attempt). Build still fails: `SIMTStackEntry` / `simt_stack` not in scope
# ...
# add_catch_test(unit_barrier_verification
#     barrier/test_barrier_verification.cpp
# )
```

Replace with:
```cmake
add_catch_test(unit_barrier_verification
    barrier/test_barrier_verification.cpp
)
set_tests_properties(unit_barrier_verification PROPERTIES LABELS "unit;barrier;verification")
```

- [ ] **Step 3: Reconfigure CMake and build**

```bash
cd /workspace/project/PTX-EMU && cmake -S . -B build 2>&1 | tail -3
cd /workspace/project/PTX-EMU && cmake --build build --target unit_barrier_verification 2>&1 | tail -10
```

Expected: Build succeeds.

- [ ] **Step 4: Run the test**

```bash
cd /workspace/project/PTX-EMU/build && ctest -R "unit_barrier_verification" -V 2>&1 | tail -20
```

Expected: 1 test target reports PASS (or with documented issues).

- [ ] **Step 5: If build/run fails, report the error**

The plan assumed the SIMTStack API matches. If methods have different signatures, the test needs further fixes. Report the actual error and STOP — do not attempt to rewrite the test in this task.

- [ ] **Step 6: Commit**

```bash
cd /workspace/project/PTX-EMU && git add tests/unit/barrier/test_barrier_verification.cpp tests/unit/CMakeLists.txt && git commit -m "test(unit): re-enable unit_barrier_verification (P2-1)

Adds the missing #include \"ptxsim/simt_stack.h\" — SIMTStack
and SIMTStackEntry are defined in that header, not in
wbar.h or thread_state.h. The test code itself was correct
once the include was present. Uncomments the CMakeLists
entry that was disabled per KNOWN_ISSUES.md §D1.2."
```

---

## Task 2: P2-2 — Remove `test_cfg_debug.cpp`

**Files:**
- Delete: `tests/ptx/test_cfg_debug.cpp`
- Modify: `tests/CMakeLists.txt` (remove the disabled comment block)

- [ ] **Step 1: Verify the file is still in the tree**

```bash
ls /workspace/project/PTX-EMU/tests/ptx/test_cfg_debug.cpp
```

Expected: file exists.

- [ ] **Step 2: Remove the file**

```bash
cd /workspace/project/PTX-EMU && git rm tests/ptx/test_cfg_debug.cpp
```

- [ ] **Step 3: Remove the disabled comment block from `tests/CMakeLists.txt`**

In `tests/CMakeLists.txt` around lines 186-189, find:
```cmake
add_standalone_test(ptx/test_cfg_edge_cases.cpp)
# test_cfg_debug.cpp has pre-existing API mismatch errors (PtxVisitor::getKernels doesn't exist)
# Disabled in main branch as well
# add_standalone_test(ptx/test_cfg_debug.cpp)
# test_cfg_benchmark.cpp moved to tests/integration/cfg/integration_cfg_benchmark.cpp
```

Replace with:
```cmake
add_standalone_test(ptx/test_cfg_edge_cases.cpp)
# test_cfg_benchmark.cpp moved to tests/integration/cfg/integration_cfg_benchmark.cpp
```

- [ ] **Step 4: Verify nothing else referenced test_cfg_debug.cpp**

```bash
cd /workspace/project/PTX-EMU && grep -rn "test_cfg_debug" --include="CMakeLists.txt" --include="*.cmake" 2>&1
```

Expected: no results (the only reference was the one we removed in Step 3).

- [ ] **Step 5: Reconfigure CMake to ensure nothing breaks**

```bash
cd /workspace/project/PTX-EMU && cmake -S . -B build 2>&1 | tail -3
```

Expected: success, no errors about missing test_cfg_debug.cpp.

- [ ] **Step 6: Commit**

```bash
cd /workspace/project/PTX-EMU && git add tests/CMakeLists.txt && git commit -m "test(ptx): remove test_cfg_debug.cpp (P2-2)

The PtxVisitor API has been completely rewritten (now takes
PtxContext& in ctor, uses ANTLR4 visitor pattern, no
getKernels() method). The old standalone test was a one-off
debug tool that no longer compiles.

CFG is already covered by test_cfg_edge_cases.cpp in the
same directory. Per roadmap §4 P2-2, the cleanest fix is
removal rather than rewriting."
```

---

## Task 3: P2-3 — Rewrite `test_cc_register.cpp` to Catch2 format

**Files:**
- Modify: `tests/unit/common/test_cc_register.cpp` (full rewrite)
- Modify: `tests/unit/CMakeLists.txt` (uncomment entry)

- [ ] **Step 1: Verify the API methods used in the test exist**

```bash
grep -nE "set_cc_reg|get_carry|get_zero|get_sign|get_overflow|get_condition_codes|set_condition_codes|class ConditionCodeRegister|class ADDC|class SUBC" \
    /workspace/project/PTX-EMU/include/ptx_ir/ptx_types.h \
    /workspace/project/PTX-EMU/include/ptxsim/thread_context.h 2>&1 | head -20
```

Expected: see definitions for `ConditionCodeRegister::set_cc_reg`, `get_carry`, `ThreadContext::get_condition_codes`, `set_condition_codes`, and class `ADDC` + `SUBC`.

If any method is missing, report and STOP — the test needs API changes that are out of P2 scope.

- [ ] **Step 2: Read the current test_cc_register.cpp in full**

```bash
cat /workspace/project/PTX-EMU/tests/unit/common/test_cc_register.cpp
```

(This has already been read; use the content from the spec investigation.)

- [ ] **Step 3: Rewrite test_cc_register.cpp to Catch2 format**

Use Write tool to replace the entire file with:

```cpp
/**
 * @file test_cc_register.cpp
 * @brief Unit test (类型一) — Condition Code Register behavior with
 *        ADDC/SUBC operations and .cc qualifier.
 *
 * 3 TEST_CASEs:
 *   1. ADDC with carry flag: 255 + 1 + 1 = 1, carry=true
 *   2. SUBC with borrow flag: 10 - 15 - 1 = large, carry=true (borrow)
 *   3. ADDC without .cc qualifier: condition codes unchanged
 */
#include "catch_amalgamated.hpp"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include <cstdint>
#include <vector>

using namespace ptxsim;

TEST_CASE("CC register: ADDC with carry flag sets carry=true", "[cc][addc]") {
    ThreadContext context;

    // Set initial condition codes: carry=true, all others=false
    ConditionCodeRegister new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, true);
    new_cc_reg.set_cc_reg(ConditionCodeRegister::ZERO_INDEX, false);
    new_cc_reg.set_cc_reg(ConditionCodeRegister::SIGN_INDEX, false);
    new_cc_reg.set_cc_reg(ConditionCodeRegister::OVERFLOW_INDEX, false);
    context.set_condition_codes(new_cc_reg);

    // ADDC: 255 + 1 + 1 = 257, low 8 bits = 1
    uint8_t src1 = 255;
    uint8_t src2 = 1;
    uint8_t dst = 0;
    void *operands[3] = {&dst, &src1, &src2};

    // Build qualifiers with .cc modifier
    std::vector<Qualifier> qualifiers;
    qualifiers.push_back(Qualifier::Q_U8);
    qualifiers.push_back(Qualifier::Q_CC);

    ADDC addc_handler;
    addc_handler.processOperation(&context, operands, qualifiers);

    // Verify result and carry flag
    REQUIRE(dst == 1);
    REQUIRE(context.get_condition_codes().get_carry() == true);
}

TEST_CASE("CC register: SUBC with borrow flag sets carry=true", "[cc][subc]") {
    ThreadContext context;

    // Set initial condition codes: carry=true (as borrow flag)
    ConditionCodeRegister new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, true);
    context.set_condition_codes(new_cc_reg);

    // SUBC: 10 - 15 - 1 = -6, unsigned wraps to large; borrow since 10 < 15+1
    uint32_t src1 = 10;
    uint32_t src2 = 15;
    uint32_t dst = 0;
    void *operands[3] = {&dst, &src1, &src2};

    std::vector<Qualifier> qualifiers;
    qualifiers.push_back(Qualifier::Q_U32);
    qualifiers.push_back(Qualifier::Q_CC);

    SUBC subc_handler;
    subc_handler.processOperation(&context, operands, qualifiers);

    // Carry flag set because 10 < 15+1 (borrow occurred)
    REQUIRE(context.get_condition_codes().get_carry() == true);
}

TEST_CASE("CC register: ADDC without .cc does not update CC", "[cc][addc]") {
    ThreadContext context;

    // Set initial: carry=true
    ConditionCodeRegister new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, true);
    context.set_condition_codes(new_cc_reg);
    auto old_cc = context.get_condition_codes();

    // ADDC without .cc modifier
    uint32_t src1 = 10;
    uint32_t src2 = 20;
    uint32_t dst = 0;
    void *operands[3] = {&dst, &src1, &src2};

    std::vector<Qualifier> qualifiers;
    qualifiers.push_back(Qualifier::Q_U32);  // No Q_CC

    ADDC addc_handler;
    addc_handler.processOperation(&context, operands, qualifiers);

    // Condition codes unchanged (no .cc modifier)
    REQUIRE(context.get_condition_codes().get_carry() == old_cc.get_carry());
    REQUIRE(context.get_condition_codes().get_zero() == old_cc.get_zero());
    REQUIRE(context.get_condition_codes().get_sign() == old_cc.get_sign());
    REQUIRE(context.get_condition_codes().get_overflow() == old_cc.get_overflow());
}
```

NOTE: The plan assumes the handler API uses `processOperation` (camelCase) — verify in Step 1. If the actual API uses different method names (e.g. `process_operation` with underscore, or different parameter order), adjust the test code accordingly. The plan's intent is preserved; only method names need to match.

- [ ] **Step 4: Uncomment the `unit_cc_register` entry in CMakeLists.txt**

In `tests/unit/CMakeLists.txt` around lines 201-208, find:
```cmake
# ============================================================================
# test_cc_register.cpp has pre-existing build errors (subc_handler undeclared)
# Not built in main branch either
# add_catch_test(unit_cc_register
#     common/test_cc_register.cpp
# )
```

Replace with:
```cmake
add_catch_test(unit_cc_register
    common/test_cc_register.cpp
)
set_tests_properties(unit_cc_register PROPERTIES LABELS "unit;cc;addc;subc")
```

- [ ] **Step 5: Reconfigure CMake and build**

```bash
cd /workspace/project/PTX-EMU && cmake -S . -B build 2>&1 | tail -3
cd /workspace/project/PTX-EMU && cmake --build build --target unit_cc_register 2>&1 | tail -10
```

Expected: Build succeeds. If the handler API is different (Step 3 note), the build will fail — fix method names and retry.

- [ ] **Step 6: Run the test**

```bash
cd /workspace/project/PTX-EMU/build && ctest -R "unit_cc_register" -V 2>&1 | tail -20
```

Expected: 1 test target reports PASS (3 TEST_CASEs, ~12 assertions).

- [ ] **Step 7: If build/test fails, report**

Common issues:
- Method name casing: `process_operation` vs `processOperation` — adjust
- Missing methods on `ConditionCodeRegister` — adjust
- Missing methods on `ThreadContext` — adjust

Report the actual error and the corrected code.

- [ ] **Step 8: Commit**

```bash
cd /workspace/project/PTX-EMU && git add tests/unit/common/test_cc_register.cpp tests/unit/CMakeLists.txt && git commit -m "test(unit): re-enable unit_cc_register (P2-3)

Rewrites the test from non-Catch2 format (void test_cc_register() +
int main() + std::cout + assert) to proper Catch2 TEST_CASE format
with REQUIRE assertions. The original test was disabled in
KNOWN_ISSUES.md §D1.2 because:
  - subc_handler was undeclared (actually SUBC class exists
    at arithmetic_ext.cpp:242)
  - non-Catch2 format

The rewrite uses:
  - TEST_CASE with [cc][addc] / [cc][subc] tags
  - REQUIRE instead of assert
  - processOperation (camelCase, current API)
  - set_cc_reg / get_carry / get_condition_codes (current API)

3 TEST_CASEs cover: ADDC carry, SUBC borrow, no-.cc preservation."
```

---

## Task 4: Update `KNOWN_ISSUES.md §D1.2`

**Files:**
- Modify: `docs/developer-guide/KNOWN_ISSUES.md`

- [ ] **Step 1: Read the current §D1.2 section**

```bash
sed -n '154,210p' /workspace/project/PTX-EMU/docs/developer-guide/KNOWN_ISSUES.md
```

- [ ] **Step 2: Update each P2-1, P2-2, P2-3 entry to mark ENABLED**

For P2-1, change the first line to:
```markdown
### `unit_barrier_verification` (ENABLED 2026-06-07 — commit TBD)
```

Add a brief note: "Fix: added `#include "ptxsim/simt_stack.h"`. Was a 1-line include addition."

For P2-2, change the first line to:
```markdown
### `test_cfg_debug` (REMOVED 2026-06-07)
```

Add a brief note: "PtxVisitor API was completely rewritten; the old test was a one-off debug tool not a regression check. CFG is covered by test_cfg_edge_cases.cpp."

For P2-3, change the first line to:
```markdown
### `unit_cc_register` (ENABLED 2026-06-07 — commit TBD)
```

Add a brief note: "Rewrote to Catch2 format. SUBC class exists at arithmetic_ext.cpp:242. Original test was non-Catch2."

For P2-4 (test_wmma), keep unchanged (still disabled per roadmap).

- [ ] **Step 3: Commit**

```bash
cd /workspace/project/PTX-EMU && git add docs/developer-guide/KNOWN_ISSUES.md && git commit -m "docs: update KNOWN_ISSUES.md §D1.2 to mark P2-1/2/3 ENABLED

P2-1 (unit_barrier_verification): ENABLED — added missing include
P2-2 (test_cfg_debug): REMOVED — PtxVisitor API was rewritten
P2-3 (unit_cc_register): ENABLED — rewritten to Catch2 format
P2-4 (test_wmma): unchanged, stays disabled per roadmap"
```

---

## Task 5: Final validation

- [ ] **Step 1: Run full default sanity**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh 2>&1 | tail -5
```

Expected: `All tests passed!` exit 0. (P2-1 is in Tier 6, P2-3 is in Tier 6, no regressions.)

- [ ] **Step 2: Run the 3 new/updated tests explicitly**

```bash
cd /workspace/project/PTX-EMU/build && ctest -R "unit_barrier_verification|unit_cc_register" 2>&1 | tail -5
```

Expected: 2 test targets pass.

- [ ] **Step 3: Verify the P2 commits**

```bash
cd /workspace/project/PTX-EMU && git log --oneline -8 2>&1
```

Expected: see 4 new commits (P2-1, P2-2, P2-3, KNOWN_ISSUES update).

- [ ] **Step 4: Report**

Report:
- 4 new commit SHAs
- Test results
- Any new issues discovered
- **ROADMAP COMPLETE:** all 3 roadmap items (P1-4, P1-3, P2) done + P1-4 bug fix

---

## Self-Review Notes

- **Spec coverage:** §3.1 (P2-1) → Task 1; §3.2 (P2-2) → Task 2; §3.3 (P2-3) → Task 3; §3.4 (docs) → Task 4; §5 (success) → Task 5
- **Placeholder scan:** No TBDs, all code is complete. Task 3 Step 3 has a NOTE about method names — intentional, not a placeholder
- **Type consistency:** `processOperation` (camelCase) used consistently across Tasks 1, 3. `set_cc_reg`, `get_carry`, `get_condition_codes` used consistently in Task 3.
- **Risk acknowledged:** Task 3 depends on API verification in Step 1; failure mode is clear (report and stop, not workaround).
