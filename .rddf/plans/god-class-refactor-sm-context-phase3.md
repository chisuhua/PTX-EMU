# god-class-refactor-sm-context-phase3 Implementation Plan (REVISED 2026-08-04)

> **For agentic workers:** REQUIRED SUB-SKILL: Use skill_use("execute") to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract three cohesive helper namespaces from `src/ptxsim/core/sm_context.cpp` (862 lines) so the file shrinks to ≤600 lines while preserving every byte-identical behaviour verified by the existing test suite (especially `tests/unit/sm/test_step_b_set_blocked_cycles.cpp`'s 4-branch fallback and `tests/unit/sm/test_streaming_admission.cpp`'s admission invariants).

**Architecture:** Three helper namespaces, each implemented as a **friend helper class with static methods** (`sm_block_dispatch::Access`, `sm_warp_lifecycle::Access`, optionally `sm_barrier_wrapper::Access`). `SMContext` friend-declares each `Access` class for direct private access. Public `SMContext` methods become one-line forwarders. This is a deliberate divergence from the existing `sm_reconvergence::`/`sm_cpptlm_inject::` pattern (which need no friends because they only take `WarpContext*`/providers) — the block-dispatch and warp-lifecycle members touch ~15 `SMContext` private fields each, so an `Access`-class friend boundary is the minimal-complexity option (user-approved 2026-08-04). `WarpContext` public API, `SMContext::exe_once()` signature, and `BarrierModule` internals are **frozen** — only the internal glue moves.

**Tech Stack:** C++20, CMake 3.x, Catch2, `git worktree` isolation on branch `openspec/god-class-refactor-sm-context-phase3`. Baseline worktree at `.worktrees/baseline-god-class-p3` (Checklist B).

**Verified line sites (empirical, HEAD of this worktree 2026-08-04):**
- `include/ptxsim/sm_context.h` — the ONLY header for SMContext (NOT `src/ptxsim/core/sm_context.h`, which does not exist).
- `src/CMakeLists.txt:81-83` — source registration (`sm_context.cpp`, `sm_context_reconvergence.cpp`, `sm_context_cpptlm_inject.cpp`). There is NO `src/ptxsim/core/CMakeLists.txt`.
- `tests/unit/CMakeLists.txt:367-372` — `add_catch_test(unit_sm_step_b_set_blocked_cycles ...)` + `set_tests_properties(... LABELS "unit;sm;ptx6;step_b;injection")` is the registration pattern to copy.
- BUG-001 comment block `src/ptxsim/core/sm_context.cpp:354-359`, `w->update_active_mask()` call at `:362` — inside `exe_once()` (frozen; comment stays put).
- Extraction ranges (all verified): `add_block` 130-204, `try_admit_pending_blocks` 206-258, `cleanup_finished_blocks` 628-643, `free_shared_memory` 645-665, `reserve_resources` 667-689, `release_resources` 691-695, `get_active_warps_count` 562-570, `get_active_threads_count` 572-580, `update_state` 586-626, `select_next_group` 831-855, `suspend_and_switch` 856-862 (EOF).
- **Placeholder warning**: `select_next_group` (831-855) always returns 0 (sequential/interleaved/shortest-first all return 0) and `suspend_and_switch` (856-862) is a no-op PTX_DEBUG_EMU body — extraction is mechanical, zero behavior risk.
- Fresh `WarpContext` ctor sets `active_count=0` → a newly registered warp is NOT active. `get_active_warps_count()==0` after a plain `add_block` is the correct ground-truth assertion.
- `WarpContext::is_active()` = `active_count > 0` (`include/ptxsim/warp_context.h:78`); `update_active_mask()` syncs `active_count` from `warp_state` (T2-1 contract).

---

## File Structure

### Production Code

| File | Responsibility |
|---|---|
| `src/ptxsim/core/sm_context.cpp` | Shrink 862 → ≤600 lines. CTA-block-dispatch members become `sm_block_dispatch::Access::` calls; warp-lifecycle members become `sm_warp_lifecycle::Access::` calls; SM barrier glue (go/no-go in Task 4) becomes `sm_barrier_wrapper::Access::` calls |
| `include/ptxsim/sm_context.h` | Forward-declare the three helper namespaces; `friend class sm_block_dispatch::Access;` (+ warp_lifecycle, + barrier_wrapper if created). Public method signatures unchanged — bodies become one-line forwards (declarations already exist) |
| `src/ptxsim/core/sm_block_dispatch.{h,cpp}` | New — `sm_block_dispatch::Access` class owning `add_block`, `try_admit_pending_blocks`, `cleanup_finished_blocks`, `free_shared_memory`, `reserve_resources`, `release_resources` |
| `src/ptxsim/core/sm_warp_lifecycle.{h,cpp}` | New — `sm_warp_lifecycle::Access` class owning `update_state`, `select_next_group`, `suspend_and_switch`, `get_active_warps_count`, `get_active_threads_count` |
| `src/ptxsim/core/sm_barrier_wrapper.{h,cpp}` | New (only if Task 4 go-decision) — `sm_barrier_wrapper::Access` owning SM-level barrier glue; go/no-go decision in Task 4 |
| `src/CMakeLists.txt` | Register new `.cpp` files after line 83 (`ptxsim/core/sm_context_cpptlm_inject.cpp`) |
| `docs/roadmap/post-phase3-debt-roadmap.md` | Update §1.2 C-2 helper cap from ≤4 to ≤5; point to follow-up `exe-once-decomposition` |
| `src/ptxsim/core/AGENTS.md` | (If helper files created) Update table to include the new files |

### Tests

| File | Responsibility |
|---|---|
| `tests/unit/sm/test_sm_block_dispatch.cpp` | New — ≥3 cases with REAL assertions (admit happy path / overflow → pending / cleanup preserves pending), mirroring verified `test_streaming_admission.cpp` invariants |
| `tests/unit/sm/test_sm_warp_lifecycle.cpp` | New — ≥3 cases: warp registration count / active-count ground truth (0 after registration) / `update_state` EXIT transition on empty SM |
| `tests/unit/sm/test_sm_barrier_wrapper.cpp` | New (only if Task 4 go-decision) — ≥2 cases |
| `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` | **Preserved unchanged** — 4-branch byte-identical fallback (lessons-learned §14) |
| `tests/unit/sm/test_streaming_admission.cpp` | **Preserved unchanged** — admission oracle for the block_dispatch extraction |
| `tests/integration/barrier/*`, `tests/integration/divergence/*` | Regression-only — must stay 100% green |

---

### Task 1: Phase 0 — Verify preconditions and lock baseline (read-only)

**Files:**
- Read-only audit: `src/ptxsim/core/sm_context.{cpp}`, `include/ptxsim/sm_context.h`, `tests/unit/sm/test_step_b_set_blocked_cycles.cpp`, `tests/unit/sm/test_streaming_admission.cpp`, `docs/roadmap/post-phase3-debt-roadmap.md`

- [ ] **Step 1: Verify the C-18 archive is in place**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
ls openspec/changes/archive/ | grep -E "refactor-warp-context" || echo "FAIL: C-18 not archived"
```
Expected: at least one `*-refactor-warp-context` directory present.

- [ ] **Step 2: Create the baseline worktree (Checklist B)**
```bash
cd /workspace/project/PTX-EMU
mkdir -p .worktrees
git worktree add .worktrees/baseline-god-class-p3 main
```
Expected: a new worktree on `main`, separate from the implementation worktree. Revert-target if everything blows up.

- [ ] **Step 3: Confirm the implementation worktree already exists**
```bash
cd /workspace/project/PTX-EMU
git worktree list | grep god-class-refactor-sm-context-phase3
```
Expected: `/workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3` is listed.

- [ ] **Step 4: Record baseline line count**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
wc -l src/ptxsim/core/sm_context.cpp
```
Expected: `862 src/ptxsim/core/sm_context.cpp`.

- [ ] **Step 5: Record baseline ctest pass count**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3/build
ctest --output-on-failure 2>&1 | tee /tmp/gc_baseline_ctest.log | tail -3
grep -E "Total Test time|passed" /tmp/gc_baseline_ctest.log | tail -5
```
Expected: all tests pass; record `BASELINE_PASS=<n>` and `BASELINE_FAIL=<m>` (m=0).

- [ ] **Step 6: Verify `w->update_active_mask()` call site (verified at line 362)**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
grep -n 'w->update_active_mask()' src/ptxsim/core/sm_context.cpp
```
Expected: a match at `src/ptxsim/core/sm_context.cpp:362` (inside `exe_once()` — frozen, must NOT move).

- [ ] **Step 7: Verify the BUG-001 comment block at lines 354-359**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
sed -n '354,359p' src/ptxsim/core/sm_context.cpp
```
Expected output contains the line: `only updated by update_active_mask(). Without this fix, active_count…`

- [ ] **Step 8: Capture the current `SMContext::exe_once()` signature**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
grep -n 'SMContext::exe_once' src/ptxsim/core/sm_context.cpp include/ptxsim/sm_context.h | head -5
```
Expected: signature visible in the header. Record it as `EXE_ONCE_SIG=<exact line>` for later comparison. MUST NOT change.

- [ ] **Step 9: Verify existing helper files are present (pattern reference only — they are NOT friend-based)**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
ls src/ptxsim/core/sm_context_reconvergence.{h,cpp} src/ptxsim/core/sm_context_cpptlm_inject.{h,cpp}
grep -n "friend" include/ptxsim/sm_context.h || echo "NO FRIENDS in sm_context.h (expected — existing helpers need none)"
```
Expected: both pairs exist; `friend` grep returns nothing (confirms the existing helpers use public collaborators, NOT friends).

- [ ] **Step 10: Verify `step_b` 4-branch test exists**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
ls tests/unit/sm/test_step_b_set_blocked_cycles.cpp
grep -nE "TEST_CASE" tests/unit/sm/test_step_b_set_blocked_cycles.cpp | head -5
```
Expected: file exists with ≥4 `TEST_CASE` entries covering the 4-branch fallback.

- [ ] **Step 11: Verify `test_streaming_admission.cpp` (admission oracle) exists and passes**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3/build
ctest -R unit_sm_streaming_admission --output-on-failure 2>&1 | tail -5
```
Expected: PASS (≥3 TEST_CASEs). This file becomes the behavioral oracle for Task 2.

- [ ] **Step 12: Verify the extraction-target member signatures (MUST match these exactly)**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
grep -nE '^(bool|void|int) SMContext::' src/ptxsim/core/sm_context.cpp
```
Expected (must match the `Access` static-method signatures in Task 2/3):
```
bool SMContext::add_block(std::unique_ptr<CTAContext> block);        // 130
void SMContext::try_admit_pending_blocks();                          // 206
void SMContext::cleanup_finished_blocks();                           // 628
void SMContext::free_shared_memory(CTAContext *block);               // 645 (raw pointer!)
bool SMContext::reserve_resources(size_t shared_mem_size, int warp_count); // 667
void SMContext::release_resources(int reservation_id);               // 691
int SMContext::get_active_warps_count() const;                       // 562
int SMContext::get_active_threads_count() const;                     // 572
void SMContext::update_state();                                      // 586 (private!)
int SMContext::select_next_group(const std::vector<int> &active_lanes); // 831 (placeholder)
void SMContext::suspend_and_switch(int current_group, int next_group);  // 856 (placeholder no-op)
```

- [ ] **Step 13: No commit — Phase 0 is a read-only audit**
Continue to Task 2 only after all checks above succeed.

---

### Task 2: Phase 3 — Extract `sm_block_dispatch` (CTA admission / queue)

**Files:**
- Create: `src/ptxsim/core/sm_block_dispatch.h`, `src/ptxsim/core/sm_block_dispatch.cpp`
- Create: `tests/unit/sm/test_sm_block_dispatch.cpp`
- Modify: `include/ptxsim/sm_context.h` (friend-declare `sm_block_dispatch::Access`)
- Modify: `src/ptxsim/core/sm_context.cpp` (replace 6 member bodies with `sm_block_dispatch::Access::` calls)
- Modify: `src/CMakeLists.txt` (add `ptxsim/core/sm_block_dispatch.cpp` after line 83)
- Modify: `tests/unit/CMakeLists.txt` (register new test — copy the `add_catch_test` + `set_tests_properties` pattern at lines 367-372)

- [ ] **Step 1: Write the failing unit test (RED — REAL assertions, NOT CHECK(true))**
Create `tests/unit/sm/test_sm_block_dispatch.cpp`:
```cpp
/**
 * PTX-6 TDD: unit tests for sm_block_dispatch::Access helper namespace.
 * Covers CTA admission / pending queue / cleanup preservation.
 * Assertions mirror the verified invariants of test_streaming_admission.cpp.
 */
#include "catch_amalgamated.hpp"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/core/sm_block_dispatch.h"
#include "memory/resource_manager.h"

#include <map>
#include <memory>
#include <vector>

using namespace ptxsim;

namespace {

// Same construction helper as test_streaming_admission.cpp (lines 41-54).
std::unique_ptr<CTAContext> make_block(Dim3 blockIdx, int threads,
                                        size_t shared_mem_bytes) {
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {threads, 1, 1};
    auto block = std::make_unique<CTAContext>();
    std::vector<StatementContext> stmts;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    block->init(gridDim, blockDim, blockIdx, stmts, &name2Sym, label2pc,
                nullptr, 0, 0);
    block->sharedMemBytes = shared_mem_bytes;
    return block;
}

}  // namespace

TEST_CASE("sm_block_dispatch::Access::add_block admits a fresh CTA",
          "[unit][sm][block_dispatch]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);

    Dim3 idx = {0, 0, 0};
    auto block = make_block(idx, /*threads=*/32, /*shared_mem=*/0);
    REQUIRE(sm_block_dispatch::Access::add_block(sm, std::move(block)) == true);
    REQUIRE(sm.get_admitted_block_count() == 1);
    REQUIRE(sm.get_pending_block_count() == 0);
    REQUIRE(sm.get_total_block_count() == 1);
}

TEST_CASE("sm_block_dispatch::Access::add_block overflow → pending",
          "[unit][sm][block_dispatch]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);

    // 2-warp SM, 1-warp blocks → 2 fit, 2 must queue.
    for (int i = 0; i < 4; i++) {
        Dim3 idx = {static_cast<uint32_t>(i), 0, 0};
        auto block = make_block(idx, /*threads=*/32, /*shared_mem=*/0);
        REQUIRE(sm_block_dispatch::Access::add_block(sm, std::move(block)) == true);
    }
    REQUIRE(sm.get_admitted_block_count() == 2);
    REQUIRE(sm.get_pending_block_count() == 2);
    REQUIRE(sm.get_total_block_count() == 4);
}

TEST_CASE("sm_block_dispatch::Access::cleanup_finished_blocks preserves pending",
          "[unit][sm][block_dispatch]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);

    for (int i = 0; i < 4; i++) {
        Dim3 idx = {static_cast<uint32_t>(i), 0, 0};
        auto block = make_block(idx, /*threads=*/32, /*shared_mem=*/0);
        REQUIRE(sm_block_dispatch::Access::add_block(sm, std::move(block)));
    }
    REQUIRE(sm.get_pending_block_count() == 2);

    // Pending must NOT silently disappear when cleanup finds no finished warps.
    sm_block_dispatch::Access::cleanup_finished_blocks(sm);
    REQUIRE(sm.get_pending_block_count() == 2);
    REQUIRE(sm.get_total_block_count() == 4);
}

TEST_CASE("sm_block_dispatch::Access::add_block hard-rejects impossible blocks",
          "[unit][sm][block_dispatch][negative]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);

    // 128 threads on a 2-warp SM can NEVER fit → hard reject, no pending dump.
    Dim3 idx = {0, 0, 0};
    auto block = make_block(idx, /*threads=*/128, /*shared_mem=*/0);
    REQUIRE(sm_block_dispatch::Access::add_block(sm, std::move(block)) == false);
    REQUIRE(sm.get_total_block_count() == 0);
    REQUIRE(sm.get_admitted_block_count() == 0);
    REQUIRE(sm.get_pending_block_count() == 0);
}
```

- [ ] **Step 2: Register the test in `tests/unit/CMakeLists.txt`**
Append after the existing `add_catch_test(unit_sm_step_b_set_blocked_cycles …)` block (line 367-372):
```cmake
add_catch_test(unit_sm_block_dispatch
    sm/test_sm_block_dispatch.cpp
)
set_tests_properties(unit_sm_block_dispatch PROPERTIES LABELS "unit;sm;block_dispatch")
```

- [ ] **Step 3: Build and verify the test fails to compile/link (RED check)**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
cmake --build build -j$(nproc) 2>&1 | tee /tmp/gc_p3_build1.log | tail -10
grep -E "fatal error|undefined reference|sm_block_dispatch" /tmp/gc_p3_build1.log | head -5
```
Expected: compile error (`sm_block_dispatch.h` not found) or link error — the helper does not exist yet. This is a REAL RED (unlike the old `CHECK(true)` fake).

- [ ] **Step 4: Create `src/ptxsim/core/sm_block_dispatch.h`**
```cpp
#ifndef PTXSM_SIM_BLOCK_DISPATCH_H
#define PTXSM_SIM_BLOCK_DISPATCH_H

#include <cstddef>
#include <memory>

class SMContext;
class CTAContext;

// CTA admission / pending-queue / resource-release helpers extracted from
// SMContext (god-class-refactor-sm-context C-2 Phase 3). SMContext friends
// this Access class for direct private-member access (the block-dispatch
// members touch ~15 private fields — a friend class is the minimal-complexity
// boundary; see sm_context_reconvergence/sm_context_cpptlm_inject for the
// no-friend alternative used when only public collaborators are needed).
namespace sm_block_dispatch {

class Access {
public:
    static bool add_block(SMContext &ctx, std::unique_ptr<CTAContext> block);
    static void try_admit_pending_blocks(SMContext &ctx);
    static void cleanup_finished_blocks(SMContext &ctx);
    static void free_shared_memory(SMContext &ctx, CTAContext *block);
    static bool reserve_resources(SMContext &ctx, size_t shared_mem_size,
                                  int warp_count);
    static void release_resources(SMContext &ctx, int reservation_id);
};

}  // namespace sm_block_dispatch

#endif
```

- [ ] **Step 5: Create `src/ptxsim/core/sm_block_dispatch.cpp`**
Copy the bodies of `add_block`, `try_admit_pending_blocks`, `cleanup_finished_blocks`, `free_shared_memory`, `reserve_resources`, `release_resources` from `sm_context.cpp:130-204, 206-258, 628-643, 645-665, 667-689, 691-695` into `sm_block_dispatch::Access::add_block` etc., each taking `SMContext &ctx` as first parameter and reading/writing `ctx.<private member>` instead of the implicit member. **Do not alter the body text otherwise — lessons-learned §1 line-level diff.** In particular:
- `add_block` (130-204) MUST keep, in order: `reserve_resources` call → pending push → `shared_mem_manager_->allocate` → `release_resources` on failure → `build_shared_memory_symbol_table` → `allocated_shared_mem +=` → `physical_block_warp_counts[physical_block_id] =` → `managed_blocks.insert` → warp registration loop (`set_physical_block_id`/`set_physical_warp_id`/`set_sm_context`/`warps.push_back`/`warp_scheduler->add_warp`) → **`update_state()` call** (this is a cross-helper call: `sm_warp_lifecycle::Access::update_state(ctx)` — Task 3 defines it; until then it can be `ctx.update_state()` since the public forwarder still exists — see Task 3 Step 8 which converts it).
- `try_admit_pending_blocks` (206-258) MUST keep its internal calls to `add_block` semantics (it re-drives admission after cleanup; keep the exact flow).
- `cleanup_finished_blocks` (628-643) MUST keep the shared-memory release + `physical_block_warp_counts` bookkeeping.
- `free_shared_memory` (645-665) takes `CTAContext *block` (raw pointer, matching the public signature).
- `reserve_resources` (667-689) returns `bool` and takes `size_t shared_mem_size, int warp_count`.
- `release_resources` (691-695) takes `int reservation_id`.

- [ ] **Step 6: Update `src/CMakeLists.txt`**
Add a new line right after line 83 (`ptxsim/core/sm_context_cpptlm_inject.cpp`):
```cmake
    ptxsim/core/sm_block_dispatch.cpp
```

- [ ] **Step 7: Friend-declare `sm_block_dispatch::Access` in `SMContext`**
In `include/ptxsim/sm_context.h` (NOT `src/ptxsim/core/` — that path does not exist), add to the class declaration (e.g., at the top of the `private:` section, line ~158):
```cpp
    friend class sm_block_dispatch::Access;
```
And add the forward declaration of the namespace/class above the class (after the existing forward declarations near line 26):
```cpp
namespace sm_block_dispatch { class Access; }
```

- [ ] **Step 8: Replace the 6 member bodies with forward calls**
In `src/ptxsim/core/sm_context.cpp`, replace each of the 6 method bodies with a one-line forward:
```cpp
bool SMContext::add_block(std::unique_ptr<CTAContext> block) {
    return sm_block_dispatch::Access::add_block(*this, std::move(block));
}
void SMContext::try_admit_pending_blocks() {
    sm_block_dispatch::Access::try_admit_pending_blocks(*this);
}
void SMContext::cleanup_finished_blocks() {
    sm_block_dispatch::Access::cleanup_finished_blocks(*this);
}
void SMContext::free_shared_memory(CTAContext *block) {
    sm_block_dispatch::Access::free_shared_memory(*this, block);
}
bool SMContext::reserve_resources(size_t shared_mem_size, int warp_count) {
    return sm_block_dispatch::Access::reserve_resources(*this, shared_mem_size,
                                                        warp_count);
}
void SMContext::release_resources(int reservation_id) {
    sm_block_dispatch::Access::release_resources(*this, reservation_id);
}
```
**MUST keep the exact public signatures above** — do NOT "fix" them to match the old plan's invented `(ctx, cta)` shapes.

- [ ] **Step 9: Verify line count dropped (GREEN check)**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
wc -l src/ptxsim/core/sm_context.cpp
```
Expected: **≤712** (≥150 line net reduction from 862 baseline). If `update_state`'s body already moved with `add_block`'s forward, count may differ — the ≤712 target is computed for the 6 block_dispatch methods only.

- [ ] **Step 10: Recursive lock audit (lessons-learned §2)**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
grep -nE 'lock_guard|unique_lock' src/ptxsim/core/sm_block_dispatch.cpp src/ptxsim/core/sm_context.cpp | head -30
```
Expected: no new acquisition of the same mutex held by a public method called from a helper. If `sm_block_dispatch::Access::add_block` takes a lock, verify the public `SMContext::add_block` does NOT take the same lock (the public now just forwards — verify).

- [ ] **Step 11: Full build + ctest + step_b regression check**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
cmake --build build -j$(nproc) 2>&1 | tee /tmp/gc_p3_build2.log | tail -5
cd build && ctest --output-on-failure 2>&1 | tee /tmp/gc_p3_ctest.log | tail -5
ctest -R unit_sm_step_b_set_blocked_cycles --output-on-failure 2>&1 | tail -5
ctest -R unit_sm_block_dispatch --output-on-failure 2>&1 | tail -5
ctest -R unit_sm_streaming_admission --output-on-failure 2>&1 | tail -5
```
Expected: full build green, all 4 step_b branches PASS, all 4 new block_dispatch cases PASS, streaming admission oracle still PASS, zero regression.

- [ ] **Step 12: Verify `step_b` 4-branch test still passes (lessons-learned §14 hard lock)**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3/build
ctest -R unit_sm_step_b_set_blocked_cycles --output-on-failure -V 2>&1 | grep -E "PASS|FAIL"
```
Expected: at least 4 `PASS` lines (one per branch).

- [ ] **Step 13: Commit Phase 3**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
git add src/ptxsim/core/sm_block_dispatch.{h,cpp} \
        include/ptxsim/sm_context.h \
        src/ptxsim/core/sm_context.cpp \
        src/CMakeLists.txt \
        tests/unit/sm/test_sm_block_dispatch.cpp \
        tests/unit/CMakeLists.txt
git commit -m "refactor(sm): extract CTA block dispatch to sm_block_dispatch.{h,cpp}"
```

---

### Task 3: Phase 4 — Extract `sm_warp_lifecycle`

**Files:**
- Create: `src/ptxsim/core/sm_warp_lifecycle.h`, `src/ptxsim/core/sm_warp_lifecycle.cpp`
- Create: `tests/unit/sm/test_sm_warp_lifecycle.cpp`
- Modify: `include/ptxsim/sm_context.h` (friend-declare `sm_warp_lifecycle::Access`)
- Modify: `src/ptxsim/core/sm_context.cpp` (replace 5 member bodies; convert the cross-helper `update_state()` call in `sm_block_dispatch.cpp` if needed)
- Modify: `src/CMakeLists.txt` (add `ptxsim/core/sm_warp_lifecycle.cpp`)
- Modify: `tests/unit/CMakeLists.txt` (register new test)

- [ ] **Step 1: Write the failing unit test (RED — REAL assertions)**
Create `tests/unit/sm/test_sm_warp_lifecycle.cpp`:
```cpp
/**
 * PTX-6 TDD: unit tests for sm_warp_lifecycle::Access helper namespace.
 * Ground truth: a fresh WarpContext has active_count=0 (ctor, warp_context.cpp:174),
 * so a plain add_block registers warps that are NOT yet active.
 */
#include "catch_amalgamated.hpp"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/core/sm_warp_lifecycle.h"
#include "memory/resource_manager.h"

#include <map>
#include <memory>
#include <vector>

using namespace ptxsim;

namespace {

std::unique_ptr<CTAContext> make_block(Dim3 blockIdx, int threads,
                                        size_t shared_mem_bytes) {
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {threads, 1, 1};
    auto block = std::make_unique<CTAContext>();
    std::vector<StatementContext> stmts;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    block->init(gridDim, blockDim, blockIdx, stmts, &name2Sym, label2pc,
                nullptr, 0, 0);
    block->sharedMemBytes = shared_mem_bytes;
    return block;
}

}  // namespace

TEST_CASE("sm_warp_lifecycle::Access registers a new warp",
          "[unit][sm][warp_lifecycle]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);

    Dim3 idx = {0, 0, 0};
    auto block = make_block(idx, /*threads=*/32, /*shared_mem=*/0);
    REQUIRE(sm.add_block(std::move(block)) == true);  // public forwarder
    REQUIRE(sm.get_num_warps() == 1);
}

TEST_CASE("sm_warp_lifecycle::Access::get_active_warps_count is 0 after plain registration",
          "[unit][sm][warp_lifecycle]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);

    Dim3 idx = {0, 0, 0};
    auto block = make_block(idx, /*threads=*/32, /*shared_mem=*/0);
    REQUIRE(sm.add_block(std::move(block)) == true);

    // Fresh warps: active_count == 0 → is_active() == false.
    REQUIRE(sm_warp_lifecycle::Access::get_active_warps_count(sm) == 0);
    REQUIRE(sm_warp_lifecycle::Access::get_active_threads_count(sm) == 0);
}

TEST_CASE("sm_warp_lifecycle::Access::update_state transitions empty SM to EXIT",
          "[unit][sm][warp_lifecycle]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);

    // No warps, no managed blocks → EXIT.
    sm_warp_lifecycle::Access::update_state(sm);
    REQUIRE(sm.get_state() == EXE_STATE::EXIT);
}
```
(Adjust the third case if a fresh SMContext's default state or scheduler setup differs — verify with a quick probe build in Step 3 and fix the assertion to match observed ground truth. The invariant "no warps + no managed blocks → EXIT" is documented in the `update_state` body at sm_context.cpp:614-618.)

- [ ] **Step 2: Register the test in `tests/unit/CMakeLists.txt`**
```cmake
add_catch_test(unit_sm_warp_lifecycle
    sm/test_sm_warp_lifecycle.cpp
)
set_tests_properties(unit_sm_warp_lifecycle PROPERTIES LABELS "unit;sm;warp_lifecycle")
```

- [ ] **Step 3: Build and verify the test fails to compile/link (RED check)**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
cmake --build build -j$(nproc) 2>&1 | tee /tmp/gc_p4_build1.log | tail -10
grep -E "fatal error|undefined reference|sm_warp_lifecycle" /tmp/gc_p4_build1.log | head -5
```
Expected: compile/link error — helper does not exist yet.

- [ ] **Step 4: Create `src/ptxsim/core/sm_warp_lifecycle.h`**
```cpp
#ifndef PTXSM_SIM_WARP_LIFECYCLE_H
#define PTXSM_SIM_WARP_LIFECYCLE_H

#include <vector>

class SMContext;

// Warp registration / retirement / active-count helpers extracted from
// SMContext (god-class-refactor-sm-context C-2 Phase 4). SMContext friends
// this Access class for direct private-member access.
namespace sm_warp_lifecycle {

class Access {
public:
    static void update_state(SMContext &ctx);
    static int select_next_group(SMContext &ctx,
                                 const std::vector<int> &active_lanes);
    static void suspend_and_switch(SMContext &ctx, int current_group,
                                   int next_group);
    static int get_active_warps_count(const SMContext &ctx);
    static int get_active_threads_count(const SMContext &ctx);
};

}  // namespace sm_warp_lifecycle

#endif
```

- [ ] **Step 5: Create `src/ptxsim/core/sm_warp_lifecycle.cpp`**
Copy the bodies of `update_state` (586-626), `select_next_group` (831-855), `suspend_and_switch` (856-862), `get_active_warps_count` (562-570), `get_active_threads_count` (572-580) into `sm_warp_lifecycle::Access::` static methods, each taking `SMContext &ctx` (const for the two count getters). **Line-level diff (lessons-learned §1).** Specifically:
- `update_state` MUST keep, in order: `warp_scheduler->update_state()` → the `while` warp-removal loop (`remove_warp`, `physical_block_warp_counts[physical_block_id]--`, `warps.erase`) → **`cleanup_finished_blocks()` call** (cross-helper: `sm_block_dispatch::Access::cleanup_finished_blocks(ctx)`) → `sm_state = EXIT/RUN` decision → `stats_` update → `get_active_threads_count()` call (becomes `Access::get_active_threads_count(ctx)`).
- `select_next_group` (831-855): mechanical — returns 0 in all branches. Keep the `divergence_mode_` switch verbatim.
- `suspend_and_switch` (856-862): mechanical no-op with `PTX_DEBUG_EMU`. Keep verbatim.
- `get_active_warps_count` (562-570) / `get_active_threads_count` (572-580): const `SMContext &ctx`, iterate `ctx.warps`, call `warp->is_active()` / `warp->get_active_count()`.

- [ ] **Step 6: Update `src/CMakeLists.txt`**
Add after the `sm_block_dispatch.cpp` line:
```cmake
    ptxsim/core/sm_warp_lifecycle.cpp
```

- [ ] **Step 7: Friend-declare `sm_warp_lifecycle::Access` in `SMContext`**
In `include/ptxsim/sm_context.h`, next to the block_dispatch friend:
```cpp
    friend class sm_warp_lifecycle::Access;
```
And add the forward declaration next to the block_dispatch one:
```cpp
namespace sm_warp_lifecycle { class Access; }
```

- [ ] **Step 8: Replace the 5 member bodies with forward calls**
In `src/ptxsim/core/sm_context.cpp`:
```cpp
void SMContext::update_state() { sm_warp_lifecycle::Access::update_state(*this); }
int SMContext::select_next_group(const std::vector<int> &active_lanes) {
    return sm_warp_lifecycle::Access::select_next_group(*this, active_lanes);
}
void SMContext::suspend_and_switch(int current_group, int next_group) {
    sm_warp_lifecycle::Access::suspend_and_switch(*this, current_group, next_group);
}
int SMContext::get_active_warps_count() const {
    return sm_warp_lifecycle::Access::get_active_warps_count(*this);
}
int SMContext::get_active_threads_count() const {
    return sm_warp_lifecycle::Access::get_active_threads_count(*this);
}
```
**Cross-helper wiring**: in `sm_block_dispatch.cpp`, any `ctx.update_state()` call must now become `sm_warp_lifecycle::Access::update_state(ctx)` (include `sm_warp_lifecycle.h`). Verify with:
```bash
grep -n "update_state" src/ptxsim/core/sm_block_dispatch.cpp
```

- [ ] **Step 9: Verify `w->update_active_mask()` site and BUG-001 comment did NOT move (frozen)**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
sed -n '350,365p' src/ptxsim/core/sm_context.cpp
```
Expected: BUG-001 comment block still at ~354-359, `w->update_active_mask()` still at ~362 (inside `exe_once()`). They are NOT part of the warp-lifecycle extraction.

- [ ] **Step 10: Recursive lock audit (lessons-learned §2)**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
grep -nE 'lock_guard|unique_lock' src/ptxsim/core/sm_warp_lifecycle.cpp src/ptxsim/core/sm_context.cpp src/ptxsim/core/sm_block_dispatch.cpp | head -30
```
Expected: no new same-mutex re-acquisition across public-method → helper boundaries.

- [ ] **Step 11: Verify line count (GREEN check)**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
wc -l src/ptxsim/core/sm_context.cpp
```
Expected: **≤600** (cumulative net reduction ≥262 from 862 baseline).

- [ ] **Step 12: Full build + ctest + regression**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
cmake --build build -j$(nproc) 2>&1 | tee /tmp/gc_p4_build2.log | tail -5
cd build && ctest --output-on-failure 2>&1 | tee /tmp/gc_p4_ctest.log | tail -5
ctest -R unit_sm_warp_lifecycle --output-on-failure 2>&1 | tail -5
ctest -R unit_sm_block_dispatch --output-on-failure 2>&1 | tail -5
ctest -R unit_sm_step_b_set_blocked_cycles --output-on-failure 2>&1 | tail -5
ctest -R unit_sm_streaming_admission --output-on-failure 2>&1 | tail -5
```
Expected: all green, zero regression.

- [ ] **Step 13: Commit Phase 4**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
git add src/ptxsim/core/sm_warp_lifecycle.{h,cpp} \
        src/ptxsim/core/sm_block_dispatch.cpp \
        include/ptxsim/sm_context.h \
        src/ptxsim/core/sm_context.cpp \
        src/CMakeLists.txt \
        tests/unit/sm/test_sm_warp_lifecycle.cpp \
        tests/unit/CMakeLists.txt
git commit -m "refactor(sm): extract warp lifecycle to sm_warp_lifecycle.{h,cpp}"
```

---

### Task 4: Phase 5 — SM barrier wrapper (go/no-go)

**Files (only if go):**
- Create: `src/ptxsim/core/sm_barrier_wrapper.{h,cpp}`
- Create: `tests/unit/sm/test_sm_barrier_wrapper.cpp`
- Modify: `include/ptxsim/sm_context.h`, `src/ptxsim/core/sm_context.cpp`, `src/CMakeLists.txt`, `tests/unit/CMakeLists.txt`

- [ ] **Step 1: Go/no-go decision (MUST)**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
wc -l src/ptxsim/core/sm_context.cpp
grep -n "BarrierModule\|barrier" src/ptxsim/core/sm_context.cpp | head -20
```
If Phase 4 left SM-level barrier glue < 50 lines → **fold-back**: skip the new file; apply a small patch folding the glue into `sm_block_dispatch.cpp` (or leaving it in place if already clean), then skip Steps 2-6 and go to Step 7. Record the decision and rationale in the commit message.

- [ ] **Step 2: (if go) Write the failing unit test (RED)**
`tests/unit/sm/test_sm_barrier_wrapper.cpp` — ≥2 cases: `cta_context->get_barrier_module()` delegation / null-barrier fallback. Use real `CTAContext` + `BarrierModule` construction; assert via the public SMContext barrier path. **No `CHECK(true)`.**

- [ ] **Step 3: (if go) Create `src/ptxsim/core/sm_barrier_wrapper.{h,cpp}`** — `sm_barrier_wrapper::Access` static methods; line-level diff (lessons-learned §1).

- [ ] **Step 4: (if go) Friend-declare `sm_barrier_wrapper::Access`** in `include/ptxsim/sm_context.h` + forward declaration.

- [ ] **Step 5: (if go) Register** in `src/CMakeLists.txt` + `tests/unit/CMakeLists.txt` (`LABELS "unit;sm;barrier_wrapper"`).

- [ ] **Step 6: (if go) Verify** `ctest -R unit_sm_barrier_wrapper --output-on-failure` 2/2 PASS + full ctest green.

- [ ] **Step 7: Recursive lock audit (lessons-learned §2)** — `grep lock_guard|unique_lock` on all three helper files + sm_context.cpp; no new same-mutex re-acquisition.

- [ ] **Step 8: Full regression** — `cmake --build build && ctest --output-on-failure` 全绿.

- [ ] **Step 9: Commit** — `git commit -m "refactor(sm): extract SM barrier wrapper to sm_barrier_wrapper.{h,cpp}"` (or fold-back patch message per Step 1 decision).

---

### Task 5: Phase 6 — Final verification

- [ ] **Step 1: MUST verify line count**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
wc -l src/ptxsim/core/sm_context.cpp
```
Expected: ≤600.

- [ ] **Step 2: MUST verify no orphan `update_active_mask` / BUG-001 sites**
```bash
grep -n 'update_active_mask' src/ptxsim/core/sm_context*.cpp
grep -n 'BUG-001' src/ptxsim/core/sm_context*.cpp
```
Expected: `update_active_mask` only inside `exe_once()` (~362); BUG-001 comment only at ~354-359. No orphans in the helper files.

- [ ] **Step 3: MUST verify `exe_once` signature unchanged**
```bash
grep -c 'exe_once' include/ptxsim/sm_context.h
```
Expected: 1 (declaration count unchanged).

- [ ] **Step 4: MUST verify full ctest green**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3/build
ctest --output-on-failure 2>&1 | tail -5
```

- [ ] **Step 5: MUST verify `step_b` 4-branch test PASS (lessons-learned §14)**
```bash
ctest -R unit_sm_step_b_set_blocked_cycles --output-on-failure -V 2>&1 | grep -c "PASS"
```
Expected: ≥4.

- [ ] **Step 6: MUST verify `tests/integration/barrier/*` + `tests/integration/divergence/*` zero regression**
```bash
ctest -L integration 2>&1 | grep -E "barrier|divergence" | tail -20
```

- [ ] **Step 7: MUST update `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2`** — helper cap ≤4 → ≤5 + point to follow-up `exe-once-decomposition`.

- [ ] **Step 8: SHOULD update `src/ptxsim/core/AGENTS.md`** table to include `sm_block_dispatch` / `sm_warp_lifecycle` / `sm_barrier_wrapper` (if created).

- [ ] **Step 9: Commit final docs**
```bash
cd /workspace/project/PTX-EMU/.rddf/wt/god-class-refactor-sm-context-phase3
git add docs/roadmap/post-phase3-debt-roadmap.md src/ptxsim/core/AGENTS.md
git commit -m "docs(sm): update C-2 roadmap + core AGENTS.md after sm_context extraction"
```

---

### Task 6: Apply phase

- [ ] **Step 1: MUST run** `openspec validate god-class-refactor-sm-context-phase3 --strict`
- [ ] **Step 2: MUST commit OpenSpec artifacts** — `git add openspec/changes/god-class-refactor-sm-context-phase3/ && git commit -m "docs(openspec): god-class-refactor-sm-context-phase3 design adjustments"` (lessons-learned §6: artifacts FIRST, before archive)
- [ ] **Step 3: MUST archive after all verification** — `openspec archive god-class-refactor-sm-context-phase3 --yes`

---

## 验收

- `src/ptxsim/core/sm_context.cpp ≤ 600 行`（基线 862，净减 ≥ 262；`<250` 为 multi-change end-state，留待 follow-up `exe-once-decomposition`）
- 新增 helper：`sm_block_dispatch.{h,cpp}` + `sm_warp_lifecycle.{h,cpp}`（+ `sm_barrier_wrapper.{h,cpp}` 若 Phase 5 go）
- helper cap ≤4 → ≤5 已记录在 `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2`
- 新增 unit tests 全部带 **真实断言**（禁 `CHECK(true)`）：`test_sm_block_dispatch.cpp`（4 case）、`test_sm_warp_lifecycle.cpp`（3 case）、`test_sm_barrier_wrapper.cpp`（2 case，若 go）
- `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` 4 分支 PASS + `test_streaming_admission.cpp` 全 PASS（admission oracle）
- `tests/integration/barrier/*` + `tests/integration/divergence/*` 零回归
- 每个 Phase commit 独立可 revert（lessons-learned §3）
- ptx-lessons-learned §1 (line-level diff), §2 (recursive lock), §14 (byte-identical fallback), Checklist B 全部勾选

## 关键约束（MUST/MUST NOT）

- MUST 行级 diff（lessons-learned §1）：迁移 body 文本不改一字；`set_*`/跨模块状态调用不丢
- MUST Checklist B：baseline worktree + 每 Phase 独立 commit
- MUST §14 step_b no-op fallback 4 分支测试锁定
- MUST §2 递归锁审计：不在持锁方法内调用同锁 public 方法
- MUST NOT 改 `SMContext::exe_once()` 签名、SM/CTA/Warp 三层调用链、WarpContext public API 签名
- MUST NOT 把 BUG-001 注释块 (354-359) 或 `w->update_active_mask()` (362) 移出 `exe_once()`
- MUST NOT 引入新 `Wbar` struct（lessons-learned §14）
- MUST NOT 用 `CHECK(true)` 充当 RED 测试 — RED 必须是编译/链接失败或真实断言失败
- MUST 使用真实文件路径：`include/ptxsim/sm_context.h`（不是不存在的 `src/ptxsim/core/sm_context.h`）；`src/CMakeLists.txt`（不是不存在的 `src/ptxsim/core/CMakeLists.txt`）
- MUST 保留 public 方法签名不变（`bool add_block(std::unique_ptr<CTAContext>)`、`bool reserve_resources(size_t,int)`、`void release_resources(int)`、`void free_shared_memory(CTAContext*)` 等 — 不是旧计划虚构的 `(ctx, cta)` 形态）
- SHOULD 复用 `BarrierModule` public API（不引入新 Wbar struct）
- SHOULD Phase 5 走 fold-back path if SM barrier glue 残留 < 50 行（避免过度工程）
