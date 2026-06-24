# Test 3 Bug Reproduction 测试清理计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 11 个重叠的 Test 3（`test_nested_sync` 屏障死锁）重现文件合并为精简的 4-5 个文件，消除约 1800 行冗余代码，同时保持全部 59 个 TEST_CASE 的覆盖率。

**Architecture:** 测试文件位于 `tests/`，通过 `tests/CMakeLists.txt` 注册，`scripts/sanity.sh` 编排。冗余集中在 3 个集群：CFG 分析（3 文件→1）、屏障执行重现（3 文件→1）、全执行环境（4 文件→2）。保留方向分析、特定 bug 和 post-barrier divergence 的唯一测试。

**Tech Stack:** Catch2 测试框架, C++20, PTX-EMU ptxsim

---

### Task 1: 合并 CFG 分析集群 — 3 文件→1 文件

**Files:**
- Delete: `tests/test_barrier_reconvergence_pc.cpp`
- Modify: `tests/test3_reproduction.cpp`（将 cfg_full 的 TEST_CASE 并入）
- Remove from: `tests/CMakeLists.txt`（移除 `test_barrier_reconvergence_pc` 条目）

- [ ] **Step 1: 分析要合并的 3 个文件内容**

```python
# 文件内容摘要：
# 1. test_test3_cfg_full.cpp (1 TEST_CASE): CFG 分析 test_nested_sync，验证第二个 barrier 的 reconvergence_pc
# 2. test_barrier_reconvergence_pc.cpp (1 TEST_CASE): 几乎相同的 PTX 布局（26 条语句），相同断言
# 3. test3_reproduction.cpp (T3-CFG-01): 集成在 18 个 TEST_CASE 中的 CFG 测试
```

- [ ] **Step 2: 读取 test_test3_cfg_full.cpp 的 TEST_CASE 内容**

```bash
cd /workspace/project/PTX-EMU && cat tests/test_test3_cfg_full.cpp
```

Expected: 看到完整的 CFG 构建和 reconvergence_pc 验证。

- [ ] **Step 3: 读取 test_barrier_reconvergence_pc.cpp 确认重复**

```bash
cd /workspace/project/PTX-EMU && cat tests/test_barrier_reconvergence_pc.cpp
```

Expected: 与 test_test3_cfg_full.cpp 几乎相同的 test_nested_sync PTX 构建 + CFG 检查。

- [ ] **Step 4: 删除 test_barrier_reconvergence_pc.cpp**

```bash
cd /workspace/project/PTX-EMU
git rm tests/test_barrier_reconvergence_pc.cpp
```

- [ ] **Step 5: 从 CMakeLists.txt 移除注册**

```bash
cd /workspace/project/PTX-EMU
grep -n "test_barrier_reconvergence_pc" tests/CMakeLists.txt
```

编辑 `tests/CMakeLists.txt`，删除包含 `test_barrier_reconvergence_pc` 的行。

---

### Task 2: 合并屏障执行重现集群 — 3 文件→1 文件

**Files:**
- Delete: `tests/test_full_barrier_execution.cpp`
- Modify: `tests/test_syncthreads_test3_repro.cpp`（保留为规范的屏障重现文件）
- Remove from: `tests/CMakeLists.txt`（移除 `test_full_barrier_execution` 条目）

- [ ] **Step 1: 确认 test_full_barrier_execution.cpp 的内容已被 test_syncthreads_test3_repro.cpp 覆盖**

```bash
cd /workspace/project/PTX-EMU && cat tests/test_full_barrier_execution.cpp | head -80
```

Expected: 看到 `build_nested_sync_statements()` 函数，它与 `test_syncthreads_test3_repro.cpp` 的第一个 TEST_CASE 几乎相同。

- [ ] **Step 2: 确认 test_syncthreads_test3_repro.cpp 的覆盖范围**

```bash
cd /workspace/project/PTX-EMU && cat tests/test_syncthreads_test3_repro.cpp | head -100
```

Expected: 看到 `test_full_barrier_execution.cpp` 的所有部分加上额外的 Wbar 直接操作测试。

- [ ] **Step 3: 删除 test_full_barrier_execution.cpp**

```bash
cd /workspace/project/PTX-EMU
git rm tests/test_full_barrier_execution.cpp
```

- [ ] **Step 4: 从 CMakeLists.txt 移除注册**

```bash
cd /workspace/project/PTX-EMU
grep -n "test_full_barrier_execution" tests/CMakeLists.txt
```

Expected: 找出该行并删除。

---

### Task 3: 合并全执行环境集群 — 4 文件→2 文件

**Files:**
- Delete: `tests/test_syncthreads_test3_full_integration.cpp`（最薄的文件，仅 98 行）
- Keep: `tests/test_syncthreads_test3_isolated.cpp`（最完善：476 行，3 个 TEST_CASE）
- Keep: `tests/test_syncthreads_full_pipeline.cpp`（最全面：324 行，5 个 TEST_CASE）
- Keep: `tests/test_syncthreads_test3_full.cpp`（唯一有 SETP 谓词评估的——保留）
- Remove from: `tests/CMakeLists.txt`（移除 `test_syncthreads_test3_full_integration` 条目）

- [ ] **Step 1: 确认 test_syncthreads_test3_full_integration.cpp 的冗余性**

```bash
cd /workspace/project/PTX-EMU && cat tests/test_syncthreads_test3_full_integration.cpp
```

Expected: 仅 98 行、1 个 TEST_CASE、1 个 SECTION，被 `test_syncthreads_test3_isolated.cpp`（476 行、3 个 TEST_CASE）完全覆盖。

- [ ] **Step 2: 删除 test_syncthreads_test3_full_integration.cpp**

```bash
cd /workspace/project/PTX-EMU
git rm tests/test_syncthreads_test3_full_integration.cpp
```

- [ ] **Step 3: 从 CMakeLists.txt 移除注册**

```
查找并删除 test_syncthreads_test3_full_integration 相关行。
```

---

### Task 4: Re-enable 精简后的测试到 sanity.sh

**Files:**
- Modify: `scripts/sanity.sh:168`（取消注释 Test 3 测试）

- [ ] **Step 1: 修改 sanity.sh 取消注释 Test 3 测试组**

在 `scripts/sanity.sh` 第 168 行附近，将：
```bash
#run_regex_tests "test_syncthreads_test3" "Test 3 deadlock reproduction"
```
改为：
```bash
run_regex_tests "test_syncthreads_test3|test3_reproduction|test_test3_cfg_full|test_syncthreads_direction|test_syncthreads_full_pipeline|test_syncthreads_test3_full|test_syncthreads_test3_isolated|test_syncthreads_test3_repro" "Test 3 deadlock reproduction (consolidated)"
```

- [ ] **Step 2: 确认 test_specific_bugs_unit 已在 sanity.sh 注册**

```bash
cd /workspace/project/PTX-EMU && grep -n "test_specific_bugs_unit\|test_post_barrier_divergence" scripts/sanity.sh
```

Expected: 这两个文件已在两种模式下注册（quick 和 full）。

---

### Task 5: 验证合并后所有测试通过

- [ ] **Step 1: 构建项目**

```bash
cd /workspace/project/PTX-EMU && . env.sh && cmake --build build 2>&1 | tail -5
```

Expected: 构建成功（由于源文件删除，可能会有链接警告但无错误）。

- [ ] **Step 2: 运行所有保留的 Test 3 测试**

```bash
cd /workspace/project/PTX-EMU/build
ctest -R "test_syncthreads_test3|test3_reproduction|test_test3_cfg_full|test_syncthreads_direction|test_syncthreads_full_pipeline|test_syncthreads_test3_full|test_syncthreads_test3_isolated|test_syncthreads_test3_repro|test_specific_bugs_unit|test_post_barrier_divergence" -V 2>&1 | tail -100
```

Expected: 所有测试通过。

- [ ] **Step 3: 运行 quick sanity**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh --quick 2>&1 | tail -20
```

Expected: 0 failures。

- [ ] **Step 4: 运行完整 sanity（可能需要较长时间）**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh 2>&1 | tail -30
```

Expected: 无新增失败。删除的文件不应影响任何测试。

---

### Task 6: 提交并清理 git history

- [ ] **Step 1: 检查所有变更**

```bash
cd /workspace/project/PTX-EMU && git status
```

Expected: 显示 3 个删除文件 + CMakeLists.txt 修改 + sanity.sh 修改。

- [ ] **Step 2: 提交**

```bash
cd /workspace/project/PTX-EMU
git add tests/CMakeLists.txt scripts/sanity.sh
git add --all  # 包含删除
git commit -m "test: consolidate Test 3 nested_sync reproduction tests

- Remove redundant test_barrier_reconvergence_pc.cpp (duplicate of test_test3_cfg_full.cpp)
- Remove redundant test_full_barrier_execution.cpp (subset of test_syncthreads_test3_repro.cpp)
- Remove redundant test_syncthreads_test3_full_integration.cpp (98-line subset)
- Re-enable consolidated tests in sanity.sh
- Preserve all unique TEST_CASE coverage across 11→8 files"
```
