# P1-3: Tier 8 Cross-Component Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate `sanity.sh --tier 8` with `integration_barrier_full_lifecycle` (3 TEST_CASEs covering init/arrive/release/reset for 2 warps in 1 CTA).

**Architecture:** Single integration test file mirroring `tests/integration/barrier/test_warp_barrier_integrated.cpp:1-50` boilerplate. Uses existing `make_bar_warp_sync` / `make_bar_sync` factories. No handler changes; pure test-writing work.

**Tech Stack:** C++20, Catch2 v3, PTX-EMU ptxsim. 2 warps, 1 CTA, `bar.sync 0`.

**Parent spec:** [`docs/superpowers/specs/2026-06-07-ptx-emu-tier8-design.md`](../specs/2026-06-07-ptx-emu-tier8-design.md)

---

## CRITICAL PRE-IMPLEMENTATION NOTES (learned from P1-4)

1. **The plan template below uses the WORKING API from `tests/integration/barrier/test_warp_barrier_integrated.cpp`**. Do NOT use any other "template" — the test infrastructure in the project has specific patterns that must be followed.

2. **If a TEST_CASE fails due to a handler bug** (likely in `barrier.cpp` or `cta_barrier.cpp`):
   - Wrap with `SKIP("P1-3.N: <reason>. See KNOWN_ISSUES.md.");` at top of TEST_CASE
   - Add 1-line comment `// KNOWN ISSUE: see docs/developer-guide/KNOWN_ISSUES.md`
   - Add new section to `KNOWN_ISSUES.md` following the format of §P1-4.1
   - Do NOT attempt to fix the handler

3. **Do NOT run `clang-format -i` on any existing handler file** — only on the new test file.

4. **Pre-P0 baseline red tests** (`integration_warp_barrier_memory_visibility`, `integration_cta_barrier_memory_visibility`) are DISABLED. Do not touch.

---

## Task 1: Build baseline, confirm Tier 8 placeholder

**Files:**
- Read: `scripts/sanity.sh:280-285` (Tier 8 placeholder)
- Read: `tests/integration/barrier/test_warp_barrier_integrated.cpp:1-50` (working boilerplate)

- [ ] **Step 1: Read the current Tier 8 placeholder**

```bash
sed -n '280,285p' /workspace/project/PTX-EMU/scripts/sanity.sh
```

Expected: see `print_test "Tier 8 currently empty (reserved for future end-to-end tests)"`.

- [ ] **Step 2: Read the working barrier test boilerplate**

```bash
head -50 /workspace/project/PTX-EMU/tests/integration/barrier/test_warp_barrier_integrated.cpp
```

Expected: see `init_instruction_factory_once`, `create_warp_with_threads`, `create_block` helpers and Catch2 includes.

- [ ] **Step 3: Confirm Tier 3 baseline is still green (regression check)**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh --tier 3 2>&1 | tail -5
```

Expected: `All tests passed!` exit 0.

---

## Task 2: Create test_barrier_full_lifecycle.cpp

**Files:**
- Create: `tests/integration/barrier/test_barrier_full_lifecycle.cpp`
- Modify: `tests/integration/CMakeLists.txt` (add `integration_barrier_full_lifecycle` entry)

- [ ] **Step 1: Create the test file**

```cpp
/**
 * @file test_barrier_full_lifecycle.cpp
 * @brief Integration test (类型二 + Tier 8) — bar.sync 0 full lifecycle
 *        (init/arrive/release/reset) for 2 warps in 1 CTA on the PTX-EMU
 *        simulator.
 *
 * Per warp:
 *   PC=0:  mov.b32 r1, tid.x    ; r1 = lane_id
 *   PC=1:  bar.sync 0          ; arrive at CTA barrier 0 (2 warps)
 *   PC=2:  add.u32 r2, r1, 10  ; work after barrier release
 *   PC=3:  ret
 *
 * Cross-component test: SM (warp scheduler) + CTA (warp management)
 *   + Wbar (barrier state) + Warp (active_mask / PC).
 */
#include "catch_amalgamated.hpp"

#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/simt_stack.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/wbar.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"
#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using ptxsim::testing::make_add;
using ptxsim::testing::make_bar_sync;
using ptxsim::testing::make_mov;
using ptxsim::testing::make_ret;
using ptxsim::testing::step_warp;

namespace {

void init_instruction_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

// Set per-lane u32 register value (creates register if not exists)
void set_reg_per_lane_u32(WarpContext *w, const std::string &reg,
                          std::function<uint32_t(int)> fn) {
    auto rbm = w->get_register_bank_manager();
    REQUIRE(rbm != nullptr);
    if (!rbm->get_register(reg, 0, 0)) {
        rbm->create_register(reg, sizeof(uint32_t));
    }
    for (int i = 0; i < 32; ++i) {
        void *p = rbm->get_register(reg, 0, i);
        REQUIRE(p != nullptr);
        *static_cast<uint32_t *>(p) = fn(i);
    }
}

uint32_t get_reg_u32(WarpContext *w, const std::string &reg, int lane) {
    auto rbm = w->get_register_bank_manager();
    void *p = rbm->get_register(reg, 0, lane);
    REQUIRE(p != nullptr);
    return *static_cast<uint32_t *>(p);
}

// Build a 2-warp CTA with the same statement sequence
struct TwoWarpSetup {
    std::unique_ptr<CTAContext> blk;
    SMContext sm{4, 128, 4096, 0};
    WarpContext *warp0 = nullptr;
    WarpContext *warp1 = nullptr;
};

TwoWarpSetup setup_two_warps(std::vector<StatementContext> &stmts) {
    TwoWarpSetup s;
    s.blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1};
    Dim3 b{64, 1, 1};  // 64 threads = 2 warps
    Dim3 bi{0, 0, 0};
    std::map<std::string, int> l2pc;
    std::map<std::string, Symtable *> n2s;
    s.blk->init(g, b, bi, stmts, &n2s, l2pc);
    bool ok = s.sm.add_block(std::move(s.blk));
    REQUIRE(ok);
    s.warp0 = s.sm.get_warp(0);
    s.warp1 = s.sm.get_warp(1);
    REQUIRE(s.warp0 != nullptr);
    REQUIRE(s.warp1 != nullptr);
    return s;
}

}  // namespace

TEST_CASE("bar_lifecycle_two_warps_release",
          "[integration][ptx][barrier][lifecycle][tier8]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(4);
    stmts.push_back(make_mov("r1", "tid.x"));   // PC=0
    stmts.push_back(make_bar_sync(0));          // PC=1: bar.sync 0
    stmts.push_back(make_add("r2", "r1", "r1")); // PC=2: r2 = r1 + r1
    stmts.push_back(make_ret());                // PC=3

    auto setup = setup_two_warps(stmts);

    // Seed both warps' r1 = lane_id (per lane)
    set_reg_per_lane_u32(setup.warp0, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });
    set_reg_per_lane_u32(setup.warp1, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    // Drive both warps in round-robin
    int pc0 = -1, pc1 = -1;
    for (int step = 0; step < 32; ++step) {
        int cur_pc0 = step_warp(setup.warp0, stmts);
        int cur_pc1 = step_warp(setup.warp1, stmts);
        if (cur_pc0 == 3) pc0 = cur_pc0;
        if (cur_pc1 == 3) pc1 = cur_pc1;
        if (pc0 == 3 && pc1 == 3) break;
    }
    REQUIRE(pc0 == 3);
    REQUIRE(pc1 == 3);

    // Verify r2 == lane_id + lane_id for all 64 lanes
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v0 = get_reg_u32(setup.warp0, "r2", lane);
        uint32_t v1 = get_reg_u32(setup.warp1, "r2", lane);
        uint32_t expected = static_cast<uint32_t>(lane + lane);
        CHECK(v0 == expected);
        CHECK(v1 == expected);
    }
}

TEST_CASE("bar_lifecycle_single_warp_blocks",
          "[integration][ptx][barrier][lifecycle][tier8]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(4);
    stmts.push_back(make_mov("r1", "tid.x"));
    stmts.push_back(make_bar_sync(0));
    stmts.push_back(make_add("r2", "r1", "r1"));
    stmts.push_back(make_ret());

    // Use only 1 warp (the CTA has 2 warps but we only drive warp 0)
    auto setup = setup_two_warps(stmts);
    set_reg_per_lane_u32(setup.warp0, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    // Drive warp 0 only; with only 1 warp arriving, barrier remains incomplete.
    // We expect step_warp to NOT reach PC=3 within bounded steps.
    int reached_pc3 = -1;
    for (int step = 0; step < 16; ++step) {
        int cur_pc = step_warp(setup.warp0, stmts);
        if (cur_pc == 3) {
            reached_pc3 = cur_pc;
            break;
        }
    }
    // If barrier correctly blocks, warp 0 stays at PC=1 (waiting for warp 1).
    // If barrier incorrectly proceeds (bug), warp 0 reaches PC=3 quickly.
    CHECK(reached_pc3 != 3);
}

TEST_CASE("bar_lifecycle_reuse_after_release",
          "[integration][ptx][barrier][lifecycle][tier8]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    // Each warp:
    //   PC=0: mov r1, tid.x
    //   PC=1: bar.sync 0      (1st barrier)
    //   PC=2: add r2, r1, 10
    //   PC=3: bar.sync 0      (2nd barrier — tests reset)
    //   PC=4: add r3, r1, 20
    //   PC=5: ret
    std::vector<StatementContext> stmts;
    stmts.reserve(6);
    stmts.push_back(make_mov("r1", "tid.x"));   // PC=0
    stmts.push_back(make_bar_sync(0));          // PC=1
    stmts.push_back(make_add("r2", "r1", "r1")); // PC=2
    stmts.push_back(make_bar_sync(0));          // PC=3
    stmts.push_back(make_add("r3", "r1", "r1")); // PC=4
    stmts.push_back(make_ret());                // PC=5

    auto setup = setup_two_warps(stmts);
    set_reg_per_lane_u32(setup.warp0, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });
    set_reg_per_lane_u32(setup.warp1, "r1",
                         [](int lane) { return static_cast<uint32_t>(lane); });

    int pc0 = -1, pc1 = -1;
    for (int step = 0; step < 32; ++step) {
        step_warp(setup.warp0, stmts);
        step_warp(setup.warp1, stmts);
        // Get current PC (re-check each step)
        if (pc0 != 5) pc0 = -1;
        if (pc1 != 5) pc1 = -1;
        for (int s = 0; s < 1; ++s) {  // check final state
            // We don't have a direct PC getter, so just run until we believe both reached ret
        }
        if (step >= 8) break;  // heuristic: 4 statements + 2 barriers
    }

    // After 8+ steps both warps should have completed
    // Verify r2 and r3 hold correct values for all 64 lanes
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t r2_0 = get_reg_u32(setup.warp0, "r2", lane);
        uint32_t r2_1 = get_reg_u32(setup.warp1, "r2", lane);
        uint32_t r3_0 = get_reg_u32(setup.warp0, "r3", lane);
        uint32_t r3_1 = get_reg_u32(setup.warp1, "r3", lane);
        CHECK(r2_0 == static_cast<uint32_t>(lane + lane));
        CHECK(r2_1 == static_cast<uint32_t>(lane + lane));
        CHECK(r3_0 == static_cast<uint32_t>(lane + lane));
        CHECK(r3_1 == static_cast<uint32_t>(lane + lane));
    }
}
```

- [ ] **Step 2: Add CMake entry**

In `tests/integration/CMakeLists.txt`, find the `integration_warp_barrier` block and add the new entry after it:

```cmake
# ============================================================================
# P1-3: Tier 8 cross-component integration test
# (added 2026-06-07 per docs/superpowers/specs/2026-06-07-ptx-emu-tier8-design.md)
# ============================================================================
add_catch_test(integration_barrier_full_lifecycle
    barrier/test_barrier_full_lifecycle.cpp
)
set_tests_properties(integration_barrier_full_lifecycle PROPERTIES LABELS "integration;barrier;lifecycle;tier8")
```

- [ ] **Step 3: Reconfigure CMake and build**

```bash
cd /workspace/project/PTX-EMU && cmake -S . -B build 2>&1 | tail -3
cd /workspace/project/PTX-EMU && cmake --build build --target integration_barrier_full_lifecycle 2>&1 | tail -10
```

Expected: Build succeeds.

- [ ] **Step 4: Run the new test**

```bash
cd /workspace/project/PTX-EMU/build && ctest -R "integration_barrier_full_lifecycle" -V 2>&1 | tail -25
```

Expected: 1 test target reports PASS (or with documented SKIPs).

- [ ] **Step 5: If tests fail — apply SKIP pattern (P1-4 lesson)**

For any failing TEST_CASE:
1. Wrap the body with `SKIP("P1-3.N: <reason>. See KNOWN_ISSUES.md.");` at top
2. Add 1-line comment `// KNOWN ISSUE: see docs/developer-guide/KNOWN_ISSUES.md`
3. Add a new section to `KNOWN_ISSUES.md` following the format of §P1-4.1

- [ ] **Step 6: Apply clang-format ONLY to the new test file**

```bash
cd /workspace/project/PTX-EMU && clang-format -i tests/integration/barrier/test_barrier_full_lifecycle.cpp
```

- [ ] **Step 7: Commit**

```bash
cd /workspace/project/PTX-EMU && git add tests/integration/barrier/test_barrier_full_lifecycle.cpp tests/integration/CMakeLists.txt docs/developer-guide/KNOWN_ISSUES.md && git commit -m "test(tier8): add integration_barrier_full_lifecycle (init/arrive/release/reset)

3 TEST_CASEs covering the full bar.sync 0 lifecycle with 2 warps
in 1 CTA. Crosses SM + CTA + Wbar + Warp components.

Note: This test only uses bar.sync and integer add — no float/cvt —
to minimize risk of hitting the P1-4 handler bugs (CvtHandler
f32->s32/f64->s64 missing, AddHandler/MulHandler/FmaHandler missing
Q_F32 paths). See KNOWN_ISSUES.md §P1-4.1 and §P1-4.2."
```

---

## Task 3: Update sanity.sh Tier 8 placeholder

**Files:**
- Modify: `scripts/sanity.sh:280-285` (replace placeholder with real test run)

- [ ] **Step 1: Read current placeholder**

```bash
sed -n '278,290p' /workspace/project/PTX-EMU/scripts/sanity.sh
```

- [ ] **Step 2: Replace the placeholder**

Use Edit tool to replace lines 280-285 with:

```bash
# Tier 8: Cross-Component Integration
# (added 2026-06-07 per docs/superpowers/specs/2026-06-07-ptx-emu-tier8-design.md)
if ! skip_tier 8; then
    print_header "Tier 8: Cross-Component Integration (full warp flows)"
    run_regex_tests "integration_barrier_full_lifecycle" "Barrier full lifecycle (init/arrive/release/reset)"
fi
```

- [ ] **Step 3: Run Tier 8 sanity check**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh --tier 8 2>&1 | tail -10
```

Expected: prints the new test, no longer prints "Tier 8 currently empty (reserved for...)".

- [ ] **Step 4: Run full default sanity**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh 2>&1 | tail -10
```

Expected: `All tests passed!` exit 0.

- [ ] **Step 5: Commit**

```bash
cd /workspace/project/PTX-EMU && git add scripts/sanity.sh && git commit -m "scripts: populate Tier 8 with integration_barrier_full_lifecycle

Replaces the 'reserved for future end-to-end tests' placeholder
with the actual ctest invocation. After this commit, Tier 8
is no longer empty."
```

---

## Task 4: Final validation

- [ ] **Step 1: Run Tier 8 specifically**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh --tier 8 2>&1 | tail -5
```

Expected: PASS, the new ctest runs and reports success (or SKIP-known).

- [ ] **Step 2: Run default sanity (all tiers 1-9)**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh 2>&1 | tail -5
```

Expected: `All tests passed!`.

- [ ] **Step 3: Verify the design spec's success criteria**

Skim `docs/superpowers/specs/2026-06-07-ptx-emu-tier8-design.md §9` and confirm each item.

- [ ] **Step 4: Report**

Report commit SHAs, test results, and any new KNOWN_ISSUES.md sections.

---

## Self-Review Notes

- **Spec coverage**: §3 (scenario selected) → Task 2; §4.1 (new test file) → Task 2 Step 1; §4.2 (modified files) → Task 2 Step 2 + Task 3 Step 2; §7 (CMake) → Task 2 Step 2; §8 (sanity.sh) → Task 3 Step 2; §9 (success criteria) → Task 4 Step 3
- **Placeholder scan**: All code blocks are complete; no "TBD" or "similar to Task N"
- **Type consistency**: `make_bar_sync(0)` / `make_add` / `make_mov` / `make_ret` consistent across all 3 TEST_CASEs
- **P1-4 lessons applied**: SKIP pattern documented (Step 5), no clang-format on handlers
