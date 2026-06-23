# T2-3 — ThreadContext / WarpContext god class POD 拆分 (Tasks)

> **总工期**: 7-10 天 | **依赖**: T1-1 ✅, T2-1 ✅, integrate-barrier-module-cta-warp 必须合并

> **TDD 三阶段**: 严格遵循 AGENTS.md §TDD 流程（每步先写测试再实现）。

> **关键策略**: 拆分为 3 阶段（Phase A 拆分 → Phase B 测试更新 → Phase C 验证），每阶段独立 commit。

> **详细 plan 与 explore 报告**: 见 `docs/superpowers/plans/2026-06-23-phase3-critical-debt.md` §Task 3 + bg_7169b052 16m43s explore 报告（8 sections A-H）

---

## Phase A: 核心拆分（6 天）

### A1: 创建 7-8 个 POD 头文件（1.5 天）

**Files:** (8 new files)
- `include/ptxsim/contexts/{exec_state,register_predicate,memory_ref,program_ref,backend_links,identity,lane_mask}.h`
- `src/ptxsim/contexts/*.cpp`（8 placeholder .cpp 文件）
- `tests/unit/contexts/test_*_pod.cpp`（8 个单元测试）

- [ ] **Step 1**: 写 8 个 POD 单元测试（Red 阶段）
- [ ] **Step 2**: 实现 8 个 POD 头文件 + 占位 .cpp（Green 阶段）
- [ ] **Step 3**: 编译 + `ctest -L "contexts" -V` 全过
- [ ] **Step 4**: `git commit -m "refactor(contexts): extract 8 POD structs from ThreadContext/WarpContext (T2-3 A1)"`

### A2: WarpState 精简（0.5 天）

**Files:**
- Modify: `include/ptxsim/warp_state.h`（6 字段 → 2 字段：`threads[]` + `exec_mask`）
- New: `tests/unit/warp/test_warp_state_minimal.cpp`
- Delete: `thread_predicates`（0 引用）、`wbars` + `current_wbar_id`（已 deprecated）、`warp_pc`（0 引用）

- [ ] **Step 1**: 写 WarpState 单元测试（Red）
- [ ] **Step 2**: 修改 `warp_state.h` 精简到 2 字段
- [ ] **Step 3**: `grep -rn "thread_predicates\|warp_state\.warp_pc\|warp_state\.wbars\|warp_state\.current_wbar_id"` 全项目访问点
- [ ] **Step 4**: 逐个修复（迁移到 BarrierModule 或直接删除）
- [ ] **Step 5**: 编译 + 跑 sanity.sh --quick
- [ ] **Step 6**: `git commit -m "refactor(warp-state): reduce from 6 to 2 fields, remove dead code (T2-3 A2)"`

### A3: ThreadContext 改 facade（1.5 天）

**Files:**
- Modify: `include/ptxsim/thread_context.h`（300 → ~150 行，4 POD 持有 + 转发方法）
- Modify: `src/ptxsim/core/thread_context.cpp`（848 行，按 POD 重排方法）

- [ ] **Step 1**: 写 facade 兼容性测试（Red）
- [ ] **Step 2**: 修改 `thread_context.h` 持有 4 POD（ExecState/RegisterPredicate/Memory/ProgramRef）
- [ ] **Step 3**: 编译（可能数百错误）+ 跑测试
- [ ] **Step 4**: 全文件传播修复字段访问点
- [ ] **Step 5**: 跑 sanity.sh --quick
- [ ] **Step 6**: `git commit -m "refactor(thread): convert to 4-POD facade, keep API compatible (T2-3 A3)"`

### A4: WarpContext 改 facade（1.5 天）

**Files:**
- Modify: `include/ptxsim/warp_context.h`（274 → ~150 行，3 POD 持有 + 转发方法）
- Modify: `src/ptxsim/core/warp_context.cpp`（487 → ~250 行）

- [ ] **Step 1**: 写 facade 兼容性测试（Red）
- [ ] **Step 2**: 修改 `warp_context.h` 持有 3 POD（LaneMask/Identity/BackendLinks）
- [ ] **Step 3**: 编译 + 跑测试
- [ ] **Step 4**: 全文件传播修复
- [ ] **Step 5**: 跑 sanity.sh --quick
- [ ] **Step 6**: `git commit -m "refactor(warp): convert to 3-POD facade, keep API compatible (T2-3 A4)"`

### A5: 物理删除 EXE_STATE / wbars / thread_predicates / warp_pc（1.0 天）

**Files:**
- Modify: `include/ptxsim/thread_context.h`（删除 `EXE_STATE state` 字段，统一到 `warp_state.threads[i].status`）
- Modify: `include/ptxsim/execution_types.h`（如 EXE_STATE 0 引用则删除整个 enum）
- Modify: 全项目传播 EXE_STATE 访问点（用 `warp_state.threads[i].status` 替换）
- Delete: `wbars` + `current_wbar_id`（已迁到 BarrierModule）+ `thread_predicates` + `warp_pc`

- [ ] **Step 1**: grep 全项目 EXE_STATE 访问点
- [ ] **Step 2**: 逐个迁移到 `warp_state.threads[i].status`
- [ ] **Step 3**: 编译 + 跑测试
- [ ] **Step 4**: 物理删除字段
- [ ] **Step 5**: `git commit -m "refactor(state): unify EXE_STATE to warp_state.threads[].status, remove dead fields (T2-3 A5)"`

---

## Phase B: 测试更新（2.5 天）

### B1: 重写测试基础设施（1.0 天）

**Files:**
- Modify: `include/ptxsim/testing/warp_test_utils.h`（207 行，重写以匹配新 API）
- Modify: `include/ptxsim/testing/scheduler_utils.h`（72 行）
- Modify: `include/ptxsim/testing/assertion_utils.h`

- [ ] **Step 1**: grep `WarpContext\|ThreadContext` 全部 69 个测试文件
- [ ] **Step 2**: 逐个重写 `warp_test_utils.h` 中的 `setup_warp()` / `setup_thread()` 等
- [ ] **Step 3**: 编译 + 跑测试
- [ ] **Step 4**: `git commit -m "refactor(test-utils): rewrite warp_test_utils for POD facade (T2-3 B1)"`

### B2: 更新 69 个测试文件（1.5 天）

**Files:** 69 个 `tests/{unit,integration,e2e}/*.cpp` 文件

- [ ] **Step 1**: `grep -rln "WarpContext\|ThreadContext" tests/` 列出全部 69 个
- [ ] **Step 2**: 批量更新字段访问（如 `thread.state` → `thread.exec_state_.state` 或 `thread.state()`）
- [ ] **Step 3**: `cmake --build build` 验证全部编译通过
- [ ] **Step 4**: `ctest` 跑全量测试
- [ ] **Step 5**: `git commit -m "refactor(tests): update 69 test files for POD facade (T2-3 B2)"`

---

## Phase C: 验证（1.0 天）

### C1: 全量 ctest + sanity.sh（0.5 天）

- [ ] `cd build && ctest --output-on-failure`（应全过）
- [ ] `cd /workspace/project/PTX-EMU && ./scripts/sanity.sh`（应全过）

### C2: ASan 验证（0.3 天）

- [ ] `cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"`
- [ ] `cmake --build build-asan && ctest --test-dir build-asan -L "barrier;e2e" -V`
- [ ] 预期：无 LeakSanitizer 报告

### C3: 性能对比（0.2 天）

- [ ] 用 callgrind 对比 `step_warp` 单步开销（应基本无差异，POD facade 是 inlined）
- [ ] `git commit -m "verify(perf): POD facade vs god class no perf regression (T2-3 closure)"`

---

## Phase 3 验证门禁

- [ ] A1-A5 全部完成（5 commits）
- [ ] B1-B2 全部完成（2 commits）
- [ ] C1-C3 全部完成（1 commit）
- [ ] ctest 全过 + sanity.sh 全过 + ASan 无泄漏
- [ ] 文档：`docs/audits/HEALTH-AUDIT-2026-06-21.md` §1.2 M2/M3 标记已修复
