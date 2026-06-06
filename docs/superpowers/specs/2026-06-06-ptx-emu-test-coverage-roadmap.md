# PTX-EMU Test Coverage Roadmap

**Date**: 2026-06-06
**Status**: Draft (pending user review)
**Scope**: 3 independent work items to advance `sanity.sh` Tier 8 / Tier 3 to green and re-enable 4 commented-out tests
**Out of scope**: WMMA/MMA implementation, new PTX instruction support

---

## 1. Background

`scripts/sanity.sh` defines a 10-tier test pyramid (Tier 1 smoke → Tier 10 bench).
Three of these tiers / sub-buckets currently have known coverage gaps that this
roadmap closes:

| Item | Bucket | Current state | Target state |
|---|---|---|---|
| **P1-4** | Tier 3 (Single Instruction Tests) | Only `test_integer_arith.cpp` + `test_ld_st_shared.cpp` exist in `tests/integration/ptx/`. The `ptx` label covers 6 instruction families but only 2 are exercised. | All 7 `reference/ptx_builtin/test_ptx_*.cu` families have simulator-driven equivalents. |
| **P1-3** | Tier 8 (Cross-Component Integration) | Empty slot — `sanity.sh:280-285` prints "reserved for future end-to-end tests". | ≥1 full-warp cross-component test runs and Tier 8 stops printing "reserved". |
| **P2** | Commented-out tests in `KNOWN_ISSUES.md §D1.2` | 4 tests exist in source tree but disabled (3 in CMakeLists, 1 requires unimplemented WMMA). | 3 of 4 re-enabled and passing (WMMA remains disabled — out of scope). |

**Pre-P0 baseline red (NOT part of this roadmap)**: `integration_warp_barrier_memory_visibility` (ctest #84) and `integration_cta_barrier_memory_visibility` (#85) are DISABLED in `tests/integration/CMakeLists.txt`. Documented in `KNOWN_ISSUES.md §Pre-P0 Baseline Red`. This roadmap **does not** touch them — they remain DISABLED.

---

## 2. Item P1-4: Tier 3 simulator-driven equivalent tests

**Goal**: Write 5 simulator-driven integration tests in `tests/integration/ptx/`, one per uncovered `reference/ptx_builtin/test_ptx_*.cu` family.

**Files to create**:

| New test | Mirrors reference | Instruction families |
|---|---|---|
| `tests/integration/ptx/test_bitwise.cpp` | `test_ptx_bitwise.cu` | and/or/xor/not/shl/shr, with B32/B64 widths |
| `tests/integration/ptx/test_cvt.cpp` | `test_ptx_cvt.cu` | cvt.s32.f32, cvt.f32.s32, cvt.f64.f32, etc. |
| `tests/integration/ptx/test_float_arith.cpp` | `test_ptx_float.cu` | fadd/fsub/fmul/fdiv/ffma, F32/F64 |
| `tests/integration/ptx/test_extended.cpp` | `test_ptx_extended.cu` | addc/subc, mad/mul24, sat variants |
| `tests/integration/ptx/test_cvta.cpp` | `test_ptx_cvta.cu` | cvta.to.global / cvta.to.shared (address conversions) |

**Template**: `tests/integration/ptx/test_integer_arith.cpp` (207 lines). Pattern:
1. Include `ptxsim/testing/scheduler_utils.h` (for `step_warp`) and `instruction_helpers.h` (for `make_*` factories).
2. For each TEST_CASE: build a minimal statement sequence via `make_mov` + the target instruction + `make_ret`.
3. Set per-lane input values via `RegisterBankManager` before driving `step_warp(w, stmts)`.
4. Assert per-lane output register values.

**CMake**: Add to `tests/integration/CMakeLists.txt` with `add_catch_test(integration_ptx_<family> ...)` and `LABELS "integration;ptx;<family>"`. Naming must follow the type-prefix convention (commit `ab55e06`).

**Success criteria**:
- `ctest -L "integration;ptx"` enumerates all 7 integration_ptx_* targets (2 existing + 5 new).
- `sanity.sh --tier 3` exits 0 (currently exits 0, but with reduced coverage).
- `./tests/ptx/test_all_ptx.sh` still passes (PTX syntax tests are orthogonal).

**Estimated effort**: 1 day. ~50 lines per test once the pattern is internalized.

**Open question**: Does `instruction_helpers.h` already have `make_and`, `make_or`, `make_fadd`, `make_cvt`, `make_cvta_to_global` factories, or do new ones need to be added? Investigate during P1-4 planning. If gaps exist, add factories as a sub-task.

---

## 3. Item P1-3: Tier 8 cross-component integration tests

**Goal**: Populate the currently-empty Tier 8 bucket in `sanity.sh` with ≥1 full-warp end-to-end test that crosses **multiple simulator components** (e.g. SM + CTA + memory + barrier).

**Candidate scenarios** (3 options — user picks):

| Scenario | Components crossed | Difficulty | Lines est. |
|---|---|---|---|
| `integration_cross_sm_shared` | 2 SMs, each with 1 block, write+read global memory to verify cross-SM visibility | High (needs multi-SM harness) | 200-300 |
| `integration_kernel_launch_flow` | Full `cudaLaunchKernel` → `__cudaRegisterFatBinary` → PtxInterpreter → SM dispatch → warp exit | Very High (largest surface) | 300-500 |
| `integration_barrier_full_lifecycle` | `bar.sync` init → arrive → release → reset, with 2 warps in 1 CTA crossing the barrier | Medium (barrier is well-isolated) | 150-200 |

**Recommendation**: Start with `integration_barrier_full_lifecycle` (lowest risk, smallest surface, validates that existing Tier 6/7 barrier tests hold up at the Tier 8 integration level). Then add `integration_kernel_launch_flow` as a follow-up if time permits.

**Open questions** (must resolve during P1-3 planning):
1. Does the test harness support 2-SM `SMContext` instantiation, or must we use a single SM with 2 blocks?
2. Does the test framework already provide a `cudaLaunchKernel` integration test pattern (look in `tests/e2e/kernel/` for `e2e_*_kernel` examples)?
3. Should Tier 8 be one big "kitchen-sink" test, or several focused tests, each crossing 2-3 components?

**Success criteria**:
- `sanity.sh --tier 8` runs ≥1 ctest and reports PASS.
- `sanity.sh` (default) exits 0 with no new failures.
- `git diff scripts/sanity.sh` shows no regression in the existing Tier 1-7 test selections.

**Estimated effort**: 2-3 days. Higher if `integration_kernel_launch_flow` is selected.

---

## 4. Item P2: Re-enable 4 commented-out tests

**Goal**: Fix the 3 of 4 disabled tests in `KNOWN_ISSUES.md §D1.2` and re-enable in CMake. WMMA test (#4) is **explicitly out of scope** (requires WMMA instruction implementation).

**Sub-items** (ordered by `KNOWN_ISSUES.md` recommended sequence):

### P2-1: `unit_barrier_verification` (30-60 min)

- **Location**: Disabled at `tests/unit/CMakeLists.txt:46-50`. File: `tests/unit/barrier/test_barrier_verification.cpp:97,112,117`.
- **Blocker**: 3-4 scope references to `SIMTStackEntry` and `simt_stack` (old API). Modern code uses `ptxsim::SIMTStack` class accessed via `#include "ptxsim/simt_stack.h"`.
- **Fix**: 
  1. Add `#include "ptxsim/simt_stack.h"` if missing.
  2. Replace `simt_stack.foo()` with `warp->get_simt_stack().foo()` (or whatever accessor is current).
  3. Replace `SIMTStackEntry` references with `ptxsim::SIMTStackEntry`.
- **Success**: Test compiles, ctest target `unit_barrier_verification` passes.

### P2-2: `test_cfg_debug` (1-2 hours)

- **Location**: Disabled at `tests/CMakeLists.txt:187-189`. File: `tests/ptx/test_cfg_debug.cpp:67`.
- **Blocker**: `PtxVisitor::getKernels()` does not exist. Test was written against older API.
- **Fix**: 
  1. Search current `src/ptx_parser/ptx_visiter.h` for the renamed API (likely `getCurrentKernel()` or `getKernels()` → returns different type).
  2. Update the call site at line 67.
  3. Adjust the test's iteration to match the new return type.
- **Success**: Test compiles, ctest target `test_cfg_debug` passes (also add to Tier 4 or Tier 9 as appropriate).

### P2-3: `unit_cc_register` (2-3 hours)

- **Location**: Disabled at `tests/unit/CMakeLists.txt:201-205`. File: `tests/unit/common/test_cc_register.cpp:8-124`.
- **Blocker**: Two problems:
  1. `subc_handler` not declared (was renamed/removed in a refactor).
  2. Test is not Catch2-formatted (uses `void test_cc_register()` + `std::cout`).
- **Fix**: 
  1. Search `src/ptxsim/instructions/` for the current SUB-with-carry implementation (likely `SubHandler` extended or a new `SubcHandler`).
  2. If handler exists, fix the import. If not, this becomes a **bug fix** task (out of scope for this roadmap — escalate to a separate spec).
  3. Rewrite test body to Catch2 `TEST_CASE` format with `REQUIRE`/`CHECK` assertions.
- **Open question**: Is `subc_handler` (or its replacement) implemented in the current source? If not, this sub-item must be split: P2-3a (Catch2 rewrite) vs P2-3b (SUB-C handler implementation). Investigate before estimating.

### P2-4: `test_wmma` (SKIP)

- **Reason**: Requires WMMA instruction implementation. Documented in `src/ptxsim/instructions/AGENTS.md` as "WMMA/MMA instructions not implemented". Out of scope.

**Success criteria**:
- 3 ctest targets uncommented in their respective CMakeLists.
- `sanity.sh --quick` and `sanity.sh` (default) exit 0.
- `KNOWN_ISSUES.md §D1.2` updated to mark these 3 as ENABLED.

**Estimated effort**: 2-3 days (P2-3 may exceed estimate if SUB-C handler is missing).

---

## 5. Execution order

**Recommended order** (lowest risk / smallest first):

1. **P1-4 (Tier 3)** — 1 day, lowest risk, well-defined template, "quick win".
2. **P1-3 (Tier 8)** — 2-3 days, requires user to pick scenario. Do `integration_barrier_full_lifecycle` first.
3. **P2-1 → P2-2 → P2-3** — sequential, P2-1 is the smallest fix (30-60 min).

**Total**: 5-7 days.

**User's original message ordering** (P1-3, P1-4, P2) is also acceptable; the natural difference is "biggest first" vs "smallest first". I recommend smallest-first because:
- Each item closes a tier / re-enables a test, providing visible progress.
- P1-4 establishes the simulator-driven test pattern that may inform P1-3's design.
- P2 items are independent and can be done at any time.

---

## 6. Risks & assumptions

| Risk | Mitigation |
|---|---|
| `instruction_helpers.h` missing `make_*` factories for new instructions | Add factories as a small sub-task during P1-4. Should be ~1 hour. |
| Tier 8 `integration_kernel_launch_flow` requires harness changes not present in current test framework | Defer to follow-up roadmap; start with `integration_barrier_full_lifecycle` first. |
| `subc_handler` is genuinely missing (not just renamed) | P2-3 splits into P2-3a (Catch2 rewrite) + P2-3b (handler implementation, separate spec). |
| Enabling P2-1/P2-2/P2-3 uncovers a real bug, expanding scope to a fix | Pause P2 work, document the bug in `KNOWN_ISSUES.md`, open a separate spec for the fix. |
| Pre-P0 baseline red tests (#84, #85) interfere with Tier 8/3 verification | They are DISABLED, so ctest will skip them. No interference expected. |

---

## 7. Success criteria (overall)

- [ ] `sanity.sh` (default, Tiers 1-9) exits 0.
- [ ] `sanity.sh --tier 3` runs ≥5 new ctest targets and all pass.
- [ ] `sanity.sh --tier 8` runs ≥1 ctest target and passes (no more "reserved" message).
- [ ] 3 of 4 P2 tests enabled and passing. `test_wmma` remains DISABLED with rationale in `KNOWN_ISSUES.md`.
- [ ] `KNOWN_ISSUES.md §D1.2` updated to mark P2-1/2/3 as ENABLED.
- [ ] No regressions in existing passing tests.

---

## 8. Per-item sub-specs (future)

After this roadmap is approved, each item gets its own `docs/superpowers/specs/2026-MM-DD-ptx-emu-<item>-design.md` + implementation plan. Items are independent — they do NOT need to be spec'd in lockstep.

Order of per-item spec → plan → implementation will follow §5.
