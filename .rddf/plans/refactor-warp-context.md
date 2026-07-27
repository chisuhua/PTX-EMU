# refactor-warp-context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use skill_use("execute") to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `src/ptxsim/core/warp_context.cpp`（558 行 god-class）拆分为 3 个职责单一子模块（active_mask / simt / dispatch），API 冻结以保证消费方 sm_context.cpp 零 diff。

**Architecture:** 三个独立子文件 + warp_context.cpp 主文件（< 300 行），通过 `#include "warp_context_*.cpp"` 模式（在 warp_context.cpp 内聚合）保持单一翻译单元。set_active_mask 保持 overwrite 语义，4 处 sync_to_warp_state() 行级随迁。

**Tech Stack:** C++20, ANTLR4 (sm_context/warp_context), Catch2 测试, CMake

---

## File Structure

### Production Code

| File | Responsibility |
|------|----------------|
| `src/ptxsim/core/warp_context.cpp` | 主文件（< 300 行），执行 active mask / SIMT / dispatch 间的协调，保留 4 处 sync_to_warp_state() |
| `src/ptxsim/core/warp_context.h` | Public API 冻结（update_active_mask / check_reconvergence / get_simt_stack / get_lanes_by_pc），包含 3 个新模块头 |
| `src/ptxsim/core/warp_context_active_mask.{h,cpp}` | set_active_mask + active_count 更新 + sync_to_warp_state 编排 |
| `src/ptxsim/core/warp_context_simt.{h,cpp}` | push_simt_stack / pop_simt_stack / check_reconvergence 编排（不重抽 simt_stack.cpp 数据结构） |
| `src/ptxsim/core/warp_context_dispatch.{h,cpp}` | execute_warp_instruction 策略表化（switch/if-else → 函数指针） |
| `src/ptxsim/core/CMakeLists.txt` | 添加 3 个新源文件到 ptxsim 目标 |

### Tests

| File | Coverage |
|------|----------|
| `tests/unit/warp/test_warp_context_active_mask.cpp` | active mask helper 单测（新增） |
| `tests/integration/warp/test_warp_context_simt.cpp` | SIMT 编排集成测（用 step_warp，新增） |
| `tests/integration/warp/test_warp_context_dispatch.cpp` | dispatch 路径验证（覆盖 execute_warp_instruction） |

---

## Tasks

### Task 1: Phase 0 — Baseline

**Files:**
- Modify: worktree metadata only (no source changes)

- [ ] **Step 1: Verify worktree exists and clean**

```bash
cd /workspace/project/PTX-EMU/.rddf/wt/refactor-warp-context
git status --short | wc -l   # expect 0 (clean)
git branch --show-current    # expect openspec/refactor-warp-context
```

- [ ] **Step 2: Capture baseline line counts**

```bash
wc -l src/ptxsim/core/warp_context.cpp  # expect 558
```

- [ ] **Step 3: Capture sync_to_warp_state baseline (§1 4 站点)**

```bash
grep -nc 'sync_to_warp_state' src/ptxsim/core/warp_context.cpp  # expect 4
grep -n 'sync_to_warp_state' src/ptxsim/core/warp_context.cpp
```

- [ ] **Step 4: Capture API freeze baseline (5 站点)**

```bash
grep -n 'update_active_mask\|check_reconvergence\|get_simt_stack\|get_lanes_by_pc' src/ptxsim/core/sm_context.cpp | head -10
# expect matches :379/:461/:468/:583/:590
```

- [ ] **Step 5: Verify baseline tests pass**

```bash
. env.sh && cmake --build build -j4 --target ptxsim && cd build && ctest -L "warp;barrier;active_mask;simt;divergence" 2>&1 | grep -E "passed|failed"
```

### Task 2: Phase 1 — Extract active mask helper

**Files:**
- Create: `src/ptxsim/core/warp_context_active_mask.{h,cpp}`
- Modify: `src/ptxsim/core/warp_context.cpp` (remove inlined active mask logic)
- Modify: `src/ptxsim/core/warp_context.h` (#include new header)
- Modify: `src/ptxsim/core/CMakeLists.txt` (add warp_context_active_mask.cpp)

- [ ] **Step 1: Create warp_context_active_mask.h with interface**

```cpp
#ifndef PTXSIM_CORE_WARP_CONTEXT_ACTIVE_MASK_H
#define PTXSIM_CORE_WARP_CONTEXT_ACTIVE_MASK_H

#include <cstdint>

namespace ptxsim {

class WarpContext;  // forward decl

namespace warp_active_mask {

void set_active_mask(WarpContext* w, uint32_t mask);  // overwrite 语义（非 OR）
uint32_t get_active_mask(const WarpContext* w);
void sync_threads_to_warp_state(WarpContext* w);  // 调用 4 处 sync_to_warp_state 的统一入口
void update_active_mask(WarpContext* w);  // API 冻结入口

}  // namespace warp_active_mask
}  // namespace ptxsim
#endif
```

- [ ] **Step 2: Create warp_context_active_mask.cpp with implementation (move verbatim from warp_context.cpp)**

读 warp_context.cpp 中所有 active mask 操作（`set_active_mask`, `get_active_mask`, `update_active_mask`，以及 4 处 `sync_to_warp_state`），原样搬到 warp_context_active_mask.cpp，使用 warp_active_mask:: 命名空间包裹。

**§1 强制**：4 处 sync_to_warp_state 必须保留（grep -c 'sync_to_warp_state' = 4 验证）。

- [ ] **Step 3: Update warp_context.cpp to call helper (preserve API signatures)**

将 warp_context.cpp 中：
- `void WarpContext::set_active_mask(uint32_t mask)` 改为 `return warp_active_mask::set_active_mask(this, mask);`（API 签名不变）
- 同理 `update_active_mask` / `sync_threads_to_warp_state`

Public API 签名零变化（消费方 sm_context.cpp:379 仍然调用 `warp->update_active_mask()`，通过 wrapper 转发）。

- [ ] **Step 4: Add #include in warp_context.h**

```cpp
#include "warp_context_active_mask.h"
```

- [ ] **Step 5: Update CMakeLists.txt**

```cmake
target_sources(ptxsim PRIVATE
    warp_context_active_mask.cpp
    warp_context_active_mask.h
    # ... 已有源
)
```

- [ ] **Step 6: Build + verify sm_context.cpp zero diff**

```bash
. env.sh && cmake --build build -j4 --target ptxsim 2>&1 | tail -3
diff <(git show HEAD:src/ptxsim/core/sm_context.cpp) src/ptxsim/core/sm_context.cpp  # expect empty
```

- [ ] **Step 7: Run active_mask + ret handler tests**

```bash
cd build && ctest -L "active_mask" -V 2>&1 | grep -E "Passed|Failed"
ctest -R "ret" -V 2>&1 | grep -E "Passed|Failed"  # ret handler 依赖 set_active_mask
```

- [ ] **Step 8: Commit**

```bash
git add src/ptxsim/core/warp_context.cpp src/ptxsim/core/warp_context.h src/ptxsim/core/warp_context_active_mask.h src/ptxsim/core/warp_context_active_mask.cpp src/ptxsim/core/CMakeLists.txt
git commit -m "refactor(warp): extract active mask helper to warp_context_active_mask"
```

### Task 3: Phase 2 — Extract SIMT orchestration

**Files:**
- Create: `src/ptxsim/core/warp_context_simt.{h,cpp}`
- Modify: `src/ptxsim/core/warp_context.cpp` (remove inlined SIMT orchestration)
- Modify: `src/ptxsim/core/warp_context.h` (#include new header)
- Modify: `src/ptxsim/core/CMakeLists.txt`

- [ ] **Step 1: Create warp_context_simt.h with interface**

```cpp
#ifndef PTXSIM_CORE_WARP_CONTEXT_SIMT_H
#define PTXSIM_CORE_WARP_CONTEXT_SIMT_H

namespace ptxsim {

class WarpContext;

namespace warp_simt {

void push_simt_stack(WarpContext* w, uint32_t mask, int reconv_pc, bool is_uni);
void pop_simt_stack(WarpContext* w);
bool check_reconvergence(WarpContext* w, int target_pc);

}  // namespace warp_simt
}  // namespace ptxsim
#endif
```

- [ ] **Step 2: Create warp_context_simt.cpp by moving warp_context.cpp:64-143 verbatim**

将 warp_context.cpp:64-143（push/pop/check_reconvergence 编排逻辑）原样搬到 warp_context_simt.cpp，使用 warp_simt:: 命名空间包裹。

**§1 强制**：4 处 sync_to_warp_state 必须保留。

**约束**：simt_stack.cpp/h 数据结构零 diff（已存在独立模块）。

- [ ] **Step 3: Update warp_context.cpp to delegate**

WarpContext::push_simt_stack / pop_simt_stack / check_reconvergence 改为 wrapper：
```cpp
void WarpContext::push_simt_stack(uint32_t mask, int pc, bool uni) { return warp_simt::push_simt_stack(this, mask, pc, uni); }
```

Public API 签名零变化。

- [ ] **Step 4: Update warp_context.h to #include new module**

- [ ] **Step 5: Update CMakeLists.txt**

- [ ] **Step 6: Build + verify sm_context.cpp zero diff**

- [ ] **Step 7: Run barrier + divergence tests**

```bash
cd build && ctest -L "barrier;divergence" -V 2>&1 | grep -E "Passed|Failed"
```

- [ ] **Step 8: Commit**

```bash
git commit -m "refactor(warp): extract SIMT orchestration to warp_context_simt"
```

### Task 4: Phase 3 — Extract instruction dispatch

**Files:**
- Create: `src/ptxsim/core/warp_context_dispatch.{h,cpp}`
- Modify: `src/ptxsim/core/warp_context.cpp` (replace execute_warp_instruction body)
- Modify: `src/ptxsim/core/warp_context.h` (#include new header)
- Modify: `src/ptxsim/core/CMakeLists.txt`

- [ ] **Step 1: Create warp_context_dispatch.h with dispatch table interface**

```cpp
#ifndef PTXSIM_CORE_WARP_CONTEXT_DISPATCH_H
#define PTXSIM_CORE_WARP_CONTEXT_DISPATCH_H

#include "ptx_ir/statement_context.h"
#include <functional>
#include <unordered_map>

namespace ptxsim {

class WarpContext;

namespace warp_dispatch {

using HandlerFunc = std::function<void(WarpContext*, const ptxir::StatementContext&)>;

void execute_warp_instruction(WarpContext* w, const ptxir::StatementContext& stmt);
void register_handler(const std::string& stmt_kind, HandlerFunc handler);

}  // namespace warp_dispatch
}  // namespace ptxsim
#endif
```

- [ ] **Step 2: Create warp_context_dispatch.cpp with strategy table**

将 warp_context.cpp 中 `execute_warp_instruction` 的 switch/if-else 改为基于 `warp_dispatch::HandlerFunc` 的策略表（key 为 statement kind 字符串，value 为 handler）。

**§1 强制**：dispatch 表中的每个 handler 内 4 处 sync_to_warp_state 必须保留（如果原代码中有）。

- [ ] **Step 3: Update warp_context.cpp**

`WarpContext::execute_warp_instruction(stmt)` 改为 `return warp_dispatch::execute_warp_instruction(this, stmt);`（API 签名不变）。

- [ ] **Step 4: Update warp_context.h to #include new module**

- [ ] **Step 5: Update CMakeLists.txt**

- [ ] **Step 6: Build + verify sm_context.cpp zero diff + test-coverage-enforcer**

```bash
. env.sh && cmake --build build -j4 --target ptxsim 2>&1 | tail -3
diff <(git show HEAD:src/ptxsim/core/sm_context.cpp) src/ptxsim/core/sm_context.cpp
ctest -L "unit;integration" 2>&1 | grep -E "passed|failed"
```

- [ ] **Step 7: Commit**

```bash
git commit -m "refactor(warp): extract instruction dispatch to warp_context_dispatch"
```

### Task 5: Phase 4 — Final verification + docs

**Files:**
- Modify: `src/ptxsim/core/AGENTS.md` (document 3 new sub-modules)

- [ ] **Step 1: Verify warp_context.cpp < 300 lines**

```bash
wc -l src/ptxsim/core/warp_context.cpp  # expect < 300
```

- [ ] **Step 2: Verify sync_to_warp_state count ≥ 4 (across all warp_context_*.cpp)**

```bash
grep -c 'sync_to_warp_state' src/ptxsim/core/warp_context*.cpp | grep -v ":0"  # expect ≥ 4 total
```

- [ ] **Step 3: Verify sm_context.cpp zero diff**

- [ ] **Step 4: Run full ctest**

```bash
cd build && ctest --output-on-failure 2>&1 | grep -E "passed|failed"
```

- [ ] **Step 5: Update AGENTS.md**

在 src/ptxsim/core/AGENTS.md 添加 WarpContext sub-module layout 表（参考 split-ptx-visitor-god-class 模式）。

- [ ] **Step 6: Commit**

```bash
git commit -m "docs(warp): document 3 new warp_context sub-modules in AGENTS.md"
```

### Task 6: Phase 5 — Validate + archive

**Files:**
- Modify: `openspec/changes/refactor-warp-context/tasks.md` (mark all [x])

- [ ] **Step 1: openspec validate --strict**

```bash
openspec validate refactor-warp-context --strict
```

- [ ] **Step 2: Archive change**

```bash
openspec archive refactor-warp-context --yes
```

- [ ] **Step 3: Mark remaining tasks done + commit**

```bash
sed -i 's/^- \[ \] 6\.1/- [x] 6.1/' openspec/changes/refactor-warp-context/tasks.md
sed -i 's/^- \[ \] 6\.2/- [x] 6.2/' openspec/changes/refactor-warp-context/tasks.md
git add -A && git commit -m "chore: complete archive of refactor-warp-context (43/43 tasks)"
```

---

## 关键约束（MUST/MUST NOT）

- MUST §1 行级 diff（SKILL.md:48-77）：4 站点 :337/:345/:370/:375 列入迁移清单
- MUST Checklist B（SKILL.md:474-483）：worktree + 3 Phase commit
- MUST set_active_mask overwrite 语义（失败模式速查表 + AGENTS.md ANTI-PATTERNS；非 §2）
- MUST NOT 改 WarpContext public API 签名（消费方 sm_context.cpp:379/:461/:468/:583/:590）
- MUST NOT 改 ret handler / execute_warp_instruction 主入口
- MUST NOT 重新抽离 simt_stack.cpp 数据结构（已存在）

## 验收

- warp_context.cpp < 300 行（从 558 → < 300）
- 新组件 ≤ 3 个（active_mask / simt / dispatch）
- grep -c 'sync_to_warp_state' 合计 ≥ 4（4 站点全部保留）
- sm_context.cpp 零 diff 编译通过（API 冻结证据）
- barrier/active_mask/ret handler 测试全绿
- test-coverage-enforcer 验证 execute_warp_instruction 路径
- ptx-lessons-learned Checklist B 全部勾选
- 所有 3 个 Phase commit 独立可 revert