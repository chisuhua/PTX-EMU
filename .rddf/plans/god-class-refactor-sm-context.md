# god-class-refactor-sm-context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use skill_use("execute") to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `src/ptxsim/core/sm_context.cpp`（965 行 god-class）拆分为 3-4 个职责单一子模块 + 去重 130 行 reconvergence 循环 + 提取 ADR-0020 cpptlm 注入代码。

**Architecture:** 共享 helper (sm_context_reconvergence.{h,cpp}) + 注入模块 (sm_context_cpptlm_inject.{h,cpp}) + 拆分（CTA/scheduler + barrier/warp_lifecycle）。Public API 冻结。

**Tech Stack:** C++20, WarpContext (refactor-warp-context C-18 已冻结), BarrierModule (post-migration), Catch2 测试, CMake

---

## File Structure

### Production Code

| File | Responsibility |
|------|----------------|
| `src/ptxsim/core/sm_context.cpp` | 主文件（< 250 行），exe_once() 主循环 + 调度协调 |
| `src/ptxsim/core/sm_context_reconvergence.{h,cpp}` | `sm_reconvergence::` — 共享 reconvergence 编排 helper（去重 :455-490 / :580-623 两段） |
| `src/ptxsim/core/sm_context_cpptlm_inject.{h,cpp}` | `sm_cpptlm_inject::` — ADR-0020 step_b + 3 setter + 3-step 编排 |
| `src/ptxsim/core/CMakeLists.txt` | 添加新源文件 |

### Tests

| File | Coverage |
|------|----------|
| `tests/unit/sm/test_sm_context_reconvergence.cpp` | 共享 helper 单测（新增） |
| `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` | step_b 4 分支测试（现有 — §14 锁定） |

---

## Tasks

### Task 1: Phase 0 — Baseline (verify C-18 done + record metrics)

**Files:**
- Modify: worktree metadata only

- [ ] **Step 1: Verify C-18 (refactor-warp-context) landed**

```bash
cd /workspace/project/PTX-EMU
git log --oneline | grep -i "refactor-warp-context" | head -3
```

- [ ] **Step 2: Verify worktree clean**

```bash
cd .rddf/wt/god-class-refactor-sm-context
git status --short | wc -l  # expect 0
```

- [ ] **Step 3: Capture baseline line count**

```bash
wc -l src/ptxsim/core/sm_context.cpp  # expect 965
```

- [ ] **Step 4: Capture baseline tests**

```bash
. env.sh && cmake --build build -j4 --target ptxsim 2>&1 | tail -3
ctest -R "sm_context|step_b|barrier|active_mask" 2>&1 | grep -E "passed|failed"
```

- [ ] **Step 5: Verify sm_context.cpp:379 update_active_mask() baseline (lessons-learned §1)**

```bash
sed -n '374,380p' src/ptxsim/core/sm_context.cpp
grep -n 'update_active_mask' src/ptxsim/core/sm_context.cpp | head -5
```

- [ ] **Step 6: Verify duplicate reconvergence loops (:455-490 and :580-623)**

```bash
grep -c 'check_reconvergence' src/ptxsim/core/sm_context.cpp  # expect ≥ 2
sed -n '455,490p' src/ptxsim/core/sm_context.cpp | head -10
sed -n '580,623p' src/ptxsim/core/sm_context.cpp | head -10
```

### Task 2: Phase 1 — Dedup reconvergence loops to shared helper

**Files:**
- Create: `src/ptxsim/core/sm_context_reconvergence.{h,cpp}`
- Modify: `src/ptxsim/core/sm_context.cpp` (replace duplicated blocks)
- Modify: `src/ptxsim/core/CMakeLists.txt`

- [ ] **Step 1: Create sm_context_reconvergence.h with interface**

```cpp
#ifndef PTXSIM_CORE_SM_CONTEXT_RECONVERGENCE_H
#define PTXSIM_CORE_SM_CONTEXT_RECONVERGENCE_H

class SMContext;  // forward decl
class WarpContext;

namespace sm_reconvergence {

// Run the post-cycle check that drains the SIMT stack and updates active_mask.
// Extracted from sm_context.cpp:455-490 and :580-623 (two near-duplicate blocks).
// MUST preserve sm_context.cpp:379 update_active_mask() call site semantics.
void drain_simt_and_update_active(SMContext* sm, WarpContext* warp);

}  // namespace sm_reconvergence
#endif
```

- [ ] **Step 2: Create sm_context_reconvergence.cpp with extracted logic**

Read sm_context.cpp:455-490 and :580-623 verbatim, identify the common core, extract to drain_simt_and_update_active.

Preserve all sm_context.cpp:374 comments and sm_context.cpp:379 update_active_mask() call sites.

- [ ] **Step 3: Update sm_context.cpp to call helper**

Replace both blocks with `sm_reconvergence::drain_simt_and_update_active(this, warp);` (preserve API freeze).

- [ ] **Step 4: Update CMakeLists.txt**

```cmake
ptxsim/core/sm_context.cpp
ptxsim/core/sm_context_reconvergence.cpp
```

- [ ] **Step 5: Build + verify line count + tests**

```bash
cmake --build build -j4 --target ptxsim 2>&1 | tail -3
grep -c 'check_reconvergence' src/ptxsim/core/sm_context.cpp  # expect ≤ 2
wc -l src/ptxsim/core/sm_context.cpp  # expect < 965 (≥ -65 lines)
ctest -L "barrier;divergence" 2>&1 | grep -E "passed|failed"
```

- [ ] **Step 6: Commit**

```bash
git commit -m "refactor(sm): dedup reconvergence orchestration loops to shared helper"
```

### Task 3: Phase 2 — Extract ADR-0020 cpptlm injection code

**Files:**
- Create: `src/ptxsim/core/sm_context_cpptlm_inject.{h,cpp}`
- Modify: `src/ptxsim/core/sm_context.cpp`
- Modify: `src/ptxsim/core/CMakeLists.txt`

- [ ] **Step 1: Create sm_context_cpptlm_inject.h**

```cpp
#ifndef PTXSIM_CORE_SM_CONTEXT_CPPTLM_INJECT_H
#define PTXSIM_CORE_SM_CONTEXT_CPPTLM_INJECT_H

class SMContext;

namespace sm_cpptlm_inject {

// 3 setter helpers (state injections for CppTLM bridge): step_b blocked cycles
// + 2 supporting setters. Encapsulated from sm_context.cpp (ADR-0020 Phase 8.B).
void step_b_set_blocked_cycles(SMContext* sm);
void set_state_for_step_b(SMContext* sm);
void inject_arrive_at_cta_barrier(SMContext* sm);

// 3-step orchestration: (1) step_b blocked cycles (2) state for step_b (3) arrive.
// MUST preserve byte-identical no-op fallback (lessons-learned §14).
void run_step_b_orchestration(SMContext* sm);

}  // namespace sm_cpptlm_inject
#endif
```

- [ ] **Step 2: Create sm_context_cpptlm_inject.cpp**

Extract step_b_set_blocked_cycles + 3-step orchestration from sm_context.cpp (the ADR-0020 cpptlm injection section).

CRITICAL: 4-branch test in test_step_b_set_blocked_cycles.cpp must remain byte-identical (lessons-learned §14). The byte-level behavior of the no-op fallback must be preserved.

- [ ] **Step 3: Update sm_context.cpp to delegate**

- [ ] **Step 4: Update CMakeLists.txt**

- [ ] **Step 5: Build + run step_b 4-branch test**

```bash
cmake --build build -j4 --target step_b_set_blocked_cycles 2>&1 | tail -3
ctest -R "step_b" --output-on-failure 2>&1 | grep -E "Passed|Failed"
```

- [ ] **Step 6: Run full ctest + Commit**

```bash
ctest -L "unit;integration" 2>&1 | grep -E "passed|failed"
git commit -m "refactor(sm): extract ADR-0020 cpptlm injection code to sm_context_cpptlm_inject"
```

### Task 4: Phase 3 — CTA scheduling + SM barrier split

**Files:**
- Create: `src/ptxsim/core/sm_context_scheduler.{h,cpp}` (CTA scheduling)
- Create: `src/ptxsim/core/sm_context_barrier.{h,cpp}` (SM barrier orchestration)
- Create: `src/ptxsim/core/sm_context_warp_lifecycle.{h,cpp}` (warp lifecycle)
- Modify: `src/ptxsim/core/sm_context.cpp`
- Modify: `src/ptxsim/core/CMakeLists.txt`

- [ ] **Step 1: Create sm_context_scheduler.{h,cpp}** with CTA scheduling logic

- [ ] **Step 2: Create sm_context_barrier.{h,cpp}** with SM-level barrier orchestration (delegates to BarrierModule)

- [ ] **Step 3: Create sm_context_warp_lifecycle.{h,cpp}** with warp lifecycle (init / add / retire)

- [ ] **Step 4: Update sm_context.cpp to delegate to all 3 modules**

exe_once() main loop signature preserved.

- [ ] **Step 5: Update CMakeLists.txt**

- [ ] **Step 6: Build + verify sm_context.cpp < 250 lines**

```bash
cmake --build build -j4 --target ptxsim 2>&1 | tail -3
wc -l src/ptxsim/core/sm_context.cpp  # expect < 250
ctest -L "unit;integration;barrier;divergence" 2>&1 | grep -E "passed|failed"
```

- [ ] **Step 7: Verify WarpContext public API zero diff**

```bash
grep -n 'update_active_mask\|check_reconvergence\|get_simt_stack\|get_lanes_by_pc' src/ptxsim/core/sm_context.cpp
# expect matches :379/:461/:468/:583/:590 (or shifted line numbers, same call signatures)
```

- [ ] **Step 8: Commit**

```bash
git commit -m "refactor(sm): split CTA scheduling + SM barrier + warp lifecycle into separate modules"
```

### Task 5: Phase 4 — Final verification + docs

- [ ] **Step 1: Verify sm_context.cpp < 250 lines**

- [ ] **Step 2: Verify sm_context.cpp:379 update_active_mask() preserved (lessons-learned §1)**

- [ ] **Step 3: Verify sm_context.cpp:374 comment preserved**

- [ ] **Step 4: Verify WarpContext API call sites zero diff (signature comparison)**

- [ ] **Step 5: Run full ctest**

```bash
cd build && ctest --output-on-failure 2>&1 | grep -E "passed|failed"
```

- [ ] **Step 6: Update AGENTS.md** to document new sub-modules

- [ ] **Step 7: Commit**

```bash
git commit -m "docs(sm): document sm_context sub-module layout in AGENTS.md"
```

### Task 6: Phase 5 — Validate + archive

- [ ] **Step 1: openspec validate --strict**

```bash
openspec validate god-class-refactor-sm-context --strict
```

- [ ] **Step 2: Archive change**

```bash
openspec archive god-class-refactor-sm-context --yes
```

- [ ] **Step 3: Mark remaining tasks + commit**

```bash
sed -i 's/^- \[ \] 6\.1/- [x] 6.1/' openspec/changes/god-class-refactor-sm-context/tasks.md
sed -i 's/^- \[ \] 6\.2/- [x] 6.2/' openspec/changes/god-class-refactor-sm-context/tasks.md
git add -A && git commit -m "chore: complete archive of god-class-refactor-sm-context (43/43 tasks)"
```

---

## 关键约束（MUST/MUST NOT）

- MUST §1 行级 diff（lessons-learned SKILL.md:48-77）：sm_context.cpp:379 update_active_mask 列入迁移清单
- MUST §14 step_b no-op fallback 4 分支测试锁定（SKILL.md:409-455）
- MUST Checklist B（worktree + 3 Phase commit）
- MUST NOT 改 exe_once() 签名、SM/CTA/Warp 三层调用链、WarpContext public API 签名
- SHOULD 复用 BarrierModule API（避免重新发明 barrier 状态机）
- MUST NOT 重新抽离 BarrierModule 内部实现

## 验收

- sm_context.cpp < 250 行（965 → < 250）
- 新组件 ≤ 4 个
- check_reconvergence 调用点 ≤ 2（经 helper）
- step_b 4 分支测试 + barrier 测试 + sm_context 单测全绿
- 集成测试零回归
- 每个 Phase commit 独立可 revert