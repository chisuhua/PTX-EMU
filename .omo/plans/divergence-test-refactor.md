# Divergence 集成测试套件重构计划

## TL;DR

> **目标**: 清理 `tests/integration/divergence/` 下 6 个文件 / 28 个 TEST_CASE，删除冗余、过时、低价值测试，合并重复测试，保留已知问题文档，强化弱断言。
>
> **交付物**:
> - 4 个精简后的测试文件（原 6 个，删除 2 个）
> - 更新的 `tests/integration/CMakeLists.txt`
> - 更新的 `scripts/sanity.sh`
> - 更新的 `AGENTS.md` 及相关文档引用
>
> **预估工作量**: Medium（约 3-4 小时执行时间）
> **并行执行**: YES — 3 个实施波次 + 1 个最终审查波次
> **关键路径**: Task 1（基线）→ Task 3（删除 iso）→ Task 7（强化 standalone）→ F1-F4（最终审查）

---

## Context

### 原始请求
用户要求审计 `tests/integration/divergence/` 下的测试覆盖情况，找出重复/过时/多余的测试，给出改进建议。

### 分析发现的关键问题
1. **`test_shortest_path_first.cpp`** — 设置 ShortestFirst 模式但从未验证实际调度行为（无分歧执行）
2. **`test_divergence_sync_isolated.cpp`** — 与 `standalone_integrated` 重复，且使用 `selp` 谓词化（无真实 SIMT 分歧）
3. **`test_post_barrier_divergence.cpp`** — 5 个 TEST_CASE 围绕同一已知 bug（`synchronize_barrier()` 未更新 `active_mask`），且 bug **尚未修复**
4. **`test_divergence_sync_convergence.cpp`** — Test A 与 C 近重复，Test D 与 E 近重复
5. **`test_nested_divergence.cpp`** — 名不副实，实际测试的是 `setp+selp` 谓词化，无嵌套分支
6. **元数据测试泛滥** — 3 个文件各含"结构验证"+"Handler 注册"共 6 个低价值 TEST_CASE

### 重构风格规范：`test_divergence_sync_convergence.cpp` 模式
本计划要求所有保留/重构的测试统一采用 `test_divergence_sync_convergence.cpp` 的构建与执行风格：

**指令构建**（`include/ptxsim/testing/instruction_helpers.h`）：
```cpp
using namespace ptxsim::testing;
auto v = build_instrs(l2pc);  // 构建 StatementContext 向量
v[BRANCH_PC] = make_bra_pred("L__BB0_4", "%p1", false, CONV_PC);
v[BRA_UNI_PC] = make_bra("L__BB0_3");
v[27] = make_ret();
```

**Predicate 设置**（`include/ptxsim/testing/predicates.h`）：
```cpp
setup_pred(w, 0x0000FFFFu);  // 设置 per-lane predicate 值
```

**执行驱动**（`include/ptxsim/testing/scheduler_utils.h`）：
```cpp
int pc = step_warp(w, v);  // 完全模拟调度器算法，返回执行的 PC
CHECK(pc == PATH_A_START);
```

**禁止直接调用 `execute_warp_instruction()`**（除非在已知问题文档测试中，如 `post_barrier` 的 BUG-REPRODUCTION）。

### Metis 审查发现
- 冗余判定标准需明确定义（代码路径覆盖 vs 断言相似性）
- `post_barrier` bug 未修复，不能"翻转"为正确行为测试
- 删除测试需有 tombstone 策略
- 外部引用遍布：`scripts/sanity.sh`、`AGENTS.md`、docs/、skills/

### Oracle Phase 1 审查
- **VERDICT: NO-GO**（初始）→ 5 个阻塞问题
- 经用户确认后已解决：改为"合并 bug-repro 为已知问题文档"，"strengthen"定义为仅重写断言
- 重新验证：CHECK [5/5] PASS → **VERDICT: GO**

### Oracle Phase 2 审查
- 计划结构和引用验证通过
- 关键发现：`synchronize_barrier()` 确实未调用 `update_active_mask()`（bug 未修复）
- **VERDICT: GO**

### Momus 高精度审查
- **VERDICT: OKAY**
- 发现：基线断言数实际为 ~207（非 ~383），已修正计划中的数字
- 所有文件引用验证存在，TEST_CASE 名称匹配，QA Scenarios 可执行

### Oracle Phase 3 审查
- 5 项检查全部通过
- **VERDICT: GO**

---

## Work Objectives

### Core Objective
将 `tests/integration/divergence/` 从 6 文件 / 28 TEST_CASE 重构为 4 文件 / ~11 TEST_CASE，消除冗余和误导性测试，保留所有真实分歧行为覆盖，不丢失已知问题文档。

### Concrete Deliverables
- `tests/integration/divergence/test_divergence_sync_standalone_integrated.cpp`（合并 iso，删除元数据，强化断言）
- `tests/integration/divergence/test_divergence_sync_convergence.cpp`（合并 A+C 和 D+E）
- `tests/integration/divergence/test_post_barrier_divergence.cpp`（合并 5→2，保留已知问题文档）
- `tests/integration/divergence/test_nested_divergence.cpp`（删除元数据，修正命名）
- 删除 `tests/integration/divergence/test_shortest_path_first.cpp`
- 删除 `tests/integration/divergence/test_divergence_sync_isolated.cpp`
- 更新 `tests/integration/CMakeLists.txt`
- 更新 `scripts/sanity.sh`
- 更新 `AGENTS.md` 及相关文档引用

### Definition of Done
- [x] `ctest -L "integration;divergence" -V` 100% 通过
- [x] `./scripts/sanity.sh` 零回归
- [x] 总断言数 ≥ 130（原 ~207，允许减少但不过度。预估：207 - 删除 shortest (~9) - 删除 iso (~20) - 合并收敛 (~15) - 清理元数据 (~25) - 合并 post_barrier (~6) + 强化 (~8) = ~140）
- [x] 无编译警告
- [x] 所有外部文档引用已更新或标记为已删除

### Must Have
- 保留所有真实分歧行为覆盖（SIMT stack 推送/弹出、调度器选择、汇聚阻塞）
- 保留 `post_barrier` 已知问题文档（`synchronize_barrier()` 未更新 `active_mask`）
- **所有测试采用 `test_divergence_sync_convergence.cpp` 风格构建**：使用 `ptxsim/testing` 工具（`make_nop`、`make_bra_pred`、`setup_pred`、`step_warp` 等），用 `step_warp` 驱动指令执行，替代直接调用 `execute_warp_instruction()`
- 更新 `CMakeLists.txt` 与文件删除原子同步
- 更新 `scripts/sanity.sh` 正则匹配

### Must NOT Have (Guardrails)
- **禁止**修改任何 `src/ptxsim/**` 源码
- **禁止**添加新的 `StatementFactory` helper
- **禁止**新增 TEST_CASE（仅重写/合并/删除现有）
- **禁止**触碰 E2E 测试（`tests/e2e/**`）
- **禁止**触碰 `integration_barrier_divergence_scheduling`（在 `barrier/` 目录）
- **禁止**重命名 `tests/integration/divergence/` 目录
- **禁止**重命名现有 ctest 目标名（除非删除对应目标）
- **禁止**修改 `.opencode/skills/` 目录下的技能文件（仅记录需要更新的引用）

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES（Catch2 + ctest + sanity.sh）
- **Automated tests**: Tests-after（重构后运行现有测试验证）
- **Framework**: Catch2（`tests/catch_amalgamated.hpp`）
- **Agent QA**: 每个任务执行后运行对应 ctest 目标，验证通过

### QA Policy
每个任务必须包含 agent-executed QA scenarios：
- **编译验证**: `cmake --build build --target <ctest目标>` 零错误
- **测试验证**: `ctest -R <目标名> -V` 100% 通过
- **回归验证**: `ctest -L "integration;divergence"` 全部通过
- **sanity 验证**: `./scripts/sanity.sh` 最终波次执行
- **证据保存**: `.omo/evidence/task-{N}-{scenario-slug}.log`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — 建立基线):
└── Task 1: 记录当前基线（断言数、通过状态、文件清单）

Wave 2 (Major Surgery — 5 个并行任务):
├── Task 2: 删除 test_shortest_path_first.cpp 及所有引用
├── Task 3: 删除 test_divergence_sync_isolated.cpp，清理 standalone 元数据
├── Task 4: 合并 test_post_barrier_divergence.cpp (5→2 TEST_CASE)
├── Task 5: 合并 test_divergence_sync_convergence.cpp (A+C, D+E)
└── Task 6: 清理 test_nested_divergence.cpp（删除元数据，修正命名）

Wave 3 (Strengthen & Docs — 2 个并行任务):
├── Task 7: 强化 standalone_integrated.cpp "all three modes" 断言
└── Task 8: 更新所有外部文档引用（AGENTS.md、sanity.sh、skills 等）

Wave FINAL (After ALL — 4 个并行审查):
├── F1: 计划合规审计 (oracle)
├── F2: 代码质量审查 (unspecified-high)
├── F3: 回归测试执行 (unspecified-high)
└── F4: 范围保真检查 (deep)
→ 汇总结果 → 获取用户显式确认

Critical Path: Task 1 → Task 3 → Task 7 → F1-F4 → user okay
Parallel Speedup: ~60%（Wave 2 的 5 个任务可完全并行）
```

### Dependency Matrix

| Task | Blocks | Blocked By |
|------|--------|-----------|
| 1 | 2,3,4,5,6 | — |
| 2 | — | 1 |
| 3 | 7 | 1 |
| 4 | — | 1 |
| 5 | — | 1 |
| 6 | — | 1 |
| 7 | — | 3 |
| 8 | — | 2,3,4,5,6 |
| F1-F4 | — | 7,8 |

---

## TODOs

- [x] 1. **记录当前基线**

  **What to do**:
  - 运行 `cd build && ctest -L "integration;divergence" -V` 记录当前 6 个目标的通过状态和断言数
  - 记录每个文件的 TEST_CASE 数和断言数（参考值，以实际 ctest 输出为准）：
    - `test_shortest_path_first.cpp`: 4 TEST_CASE, ~9 assertions
    - `test_divergence_sync_isolated.cpp`: 4 TEST_CASE, ~20 assertions
    - `test_divergence_sync_convergence.cpp`: 5 TEST_CASE, ~82 assertions
    - `test_divergence_sync_standalone_integrated.cpp`: 6 TEST_CASE, ~49 assertions
    - `test_nested_divergence.cpp`: 4 TEST_CASE, ~32 assertions
    - `test_post_barrier_divergence.cpp`: 5 TEST_CASE, 15 assertions
  - 预估总计：28 TEST_CASE, ~207 assertions
  - **重要**：以 `ctest -V` 实际输出为准，上述数字为参考值
  - 保存到 `.omo/evidence/task-1-baseline.md`

  **Must NOT do**:
  - 不要修改任何文件
  - 不要运行超出 divergence 标签的测试

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: 纯信息收集，无需专业技能

  **Parallelization**:
  - **Can Run In Parallel**: NO（所有后续任务依赖此基线）
  - **Blocks**: Tasks 2-9
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] 文件 `.omo/evidence/task-1-baseline.md` 存在且包含完整数据
  - [ ] `ctest -L "integration;divergence" -V` 输出显示 6/6 目标通过

  **QA Scenarios**:
  ```
  Scenario: 验证基线记录完整
    Tool: Bash
    Steps:
      1. cd build && ctest -L "integration;divergence" -V 2>&1 | tee /tmp/baseline.log
      2. grep -c "All tests passed" /tmp/baseline.log
    Expected Result: count == 6
    Evidence: .omo/evidence/task-1-baseline.log
  ```

  **Commit**: NO（合并到最终 commit）

---

- [x] 2. **删除 test_shortest_path_first.cpp 及所有引用**

  **What to do**:
  - 删除 `tests/integration/divergence/test_shortest_path_first.cpp`
  - 从 `tests/integration/CMakeLists.txt` 删除：
    ```cmake
    add_catch_test(integration_shortest_path_first
        divergence/test_shortest_path_first.cpp
    )
    set_tests_properties(integration_shortest_path_first PROPERTIES LABELS "integration;shortest_first;divergence")
    ```
  - 更新 `scripts/sanity.sh` 行 175：移除 `test_post_barrier_divergence` 所在组中的此项（实际该正则不含 shortest_path_first，确认无影响）
  - 更新 `AGENTS.md` 行 207：从 divergence 标签表中移除 `integration_shortest_path_first` 引用
  - 更新 `workflow-state.md` 行 85：标记为已删除
  - 在 `.opencode/skills/ptx-lane-verification/SKILL.md` 中添加注释说明文件已删除

  **Must NOT do**:
  - 不要修改 `src/` 代码
  - 不要重命名其他 ctest 目标

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: 文件删除和配置更新

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 3,4,5,8 并行）
  - **Blocks**: Task 8（文档更新）
  - **Blocked By**: Task 1

  **Acceptance Criteria**:
  - [ ] 文件 `test_shortest_path_first.cpp` 不存在
  - [ ] `grep -n "shortest_path_first" tests/integration/CMakeLists.txt` 无输出
  - [ ] `cmake --build build` 成功（无编译错误）
  - [ ] `ctest -R integration_shortest_path_first` 返回 "No tests were found"

  **QA Scenarios**:
  ```
  Scenario: 验证文件和引用已清除
    Tool: Bash
    Steps:
      1. test -f tests/integration/divergence/test_shortest_path_first.cpp && echo "FAIL" || echo "PASS"
      2. grep -c "shortest_path_first" tests/integration/CMakeLists.txt
      3. cd build && cmake --build build 2>&1 | grep -i error | wc -l
    Expected Result: (1) PASS, (2) 0, (3) 0
    Evidence: .omo/evidence/task-2-deletion.log
  ```

  **Commit**: NO（合并到最终 commit）

---

- [x] 3. **删除 test_divergence_sync_isolated.cpp，清理 standalone 中的元数据**

  **What to do**:
  - 删除 `tests/integration/divergence/test_divergence_sync_isolated.cpp`
  - 从 `tests/integration/CMakeLists.txt` 删除 `integration_divergence_sync_isolated` 目标
  - 更新 `scripts/sanity.sh` 行 176：将正则 `test_divergence_sync_standalone|test_divergence_sync_isolated` 改为仅 `test_divergence_sync_standalone`
  - 在 `test_divergence_sync_standalone_integrated.cpp` 中删除元数据 TEST_CASE：
    - "divergence_sync_standalone: statement sequence structure"
    - "divergence_sync_standalone: handler registration"
  - 添加 tombstone 注释：
    ```cpp
    // NOTE: 原 test_divergence_sync_isolated.cpp 的屏障同步测试已合并到此处。
    // 该文件使用 setp+selp 谓词化（非真实 SIMT 分歧），其屏障行为覆盖已由本文件的
    // "barrier releases all threads" 和 "full warp barrier-then-divergence flow" 覆盖。
    ```

  **Must NOT do**:
  - 不要合并 iso 的执行测试（它与 standalone 的 "full warp barrier-then-divergence flow" 重复，且使用 selp 而非真实分支）
  - 不要修改 standalone 的执行测试逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: 文件操作和简单编辑

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 2,4,5,8 并行）
  - **Blocks**: Task 7（standalone 强化）
  - **Blocked By**: Task 1

  **Acceptance Criteria**:
  - [ ] `test_divergence_sync_isolated.cpp` 不存在
  - [ ] `grep -n "divergence_sync_isolated" tests/integration/CMakeLists.txt` 无输出
  - [ ] `grep -n "divergence_sync_isolated" scripts/sanity.sh` 无输出
  - [ ] `ctest -R integration_divergence_sync_standalone -V` 通过（剩余 4 TEST_CASE）
  - [ ] standalone 文件头部有 tombstone 注释

  **QA Scenarios**:
  ```
  Scenario: 验证 iso 已删除且 standalone 仍通过
    Tool: Bash
    Steps:
      1. test ! -f tests/integration/divergence/test_divergence_sync_isolated.cpp
      2. cd build && cmake --build build
      3. ctest -R integration_divergence_sync_standalone -V 2>&1 | grep "All tests passed"
    Expected Result: (1) true, (2) 0 errors, (3) 包含 "All tests passed"
    Evidence: .omo/evidence/task-3-iso-deletion.log
  ```

  **Commit**: NO（合并到最终 commit）

---

- [x] 4. **合并 test_post_barrier_divergence.cpp (5→2 TEST_CASE)**

  **What to do**:
  - 保留 2 个 TEST_CASE，删除 3 个：
    - **保留** "BUG-REPRODUCTION: bar.warp.sync releases threads but active_mask not updated"
      - 重命名为 "KNOWN-ISSUE: bar.warp.sync releases threads but active_mask not updated"
      - 添加注释：`// KNOWN ISSUE: synchronize_barrier() does not call update_active_mask() (AGENTS.md#48)`
      - 保留所有 3 个 SECTION（Setup、BUG、Verify），它们共同构成完整的已知问题文档
    - **保留** "FIX VERIFICATION: After barrier, set_active_mask should sync with released lanes"
      - 重命名为 "WORKAROUND-VERIFY: Manual set_active_mask fixes post-barrier execution"
      - 保留 SECTION "FIX: After barrier release..."
    - **删除** "Full cycle: barrier release then post-barrier instruction execution"（与 BUG-REPRODUCTION 重复）
    - **删除** "BUG-REPRODUCTION-CTA: Post-barrier execute_warp_instruction..."（与第一个 BUG 重复，仅 setup 略有不同）
    - **删除** "FIX-VERIFICATION-CTA: After updating active_mask..."（与第二个 FIX 重复）
  - 在文件头部添加注释说明这是已知问题文档：
    ```cpp
    /**
     * KNOWN ISSUE DOCUMENTATION: Post-barrier active_mask not updated
     *
     * synchronize_barrier() (sm_context.cpp:536-637) releases threads after barrier
     * completion but does NOT call update_active_mask(). This causes execute_warp_instruction()
     * to only execute lanes that were in active_mask before the barrier.
     *
     * See: src/ptxsim/core/AGENTS.md#48
     */
    ```

  **Must NOT do**:
  - 不要翻转断言为"正确行为"（bug 未修复，翻转会导致测试失败）
  - 不要修改 `synchronize_barrier()` 实现（Scope OUT）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: 测试文件编辑，不涉及复杂逻辑

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 2,3,5,8 并行）
  - **Blocks**: Task 8（文档更新）
  - **Blocked By**: Task 1

  **Acceptance Criteria**:
  - [ ] 文件包含 2 个 TEST_CASE（原 5 个）
  - [ ] `ctest -R integration_post_barrier_divergence -V` 通过
  - [ ] 文件头部有已知问题注释
  - [ ] 保留的 TEST_CASE 名以 "KNOWN-ISSUE" 或 "WORKAROUND-VERIFY" 开头

  **QA Scenarios**:
  ```
  Scenario: 验证合并后测试通过且问题文档完整
    Tool: Bash
    Steps:
      1. grep -c "TEST_CASE" tests/integration/divergence/test_post_barrier_divergence.cpp
      2. cd build && cmake --build build
      3. ctest -R integration_post_barrier_divergence -V 2>&1 | grep "All tests passed"
    Expected Result: (1) 2, (2) 0 errors, (3) 包含 "All tests passed"
    Evidence: .omo/evidence/task-4-postbarrier-merge.log
  ```

  **Commit**: NO（合并到最终 commit）

---

- [x] 5. **合并 test_divergence_sync_convergence.cpp (A+C, D+E)**

  **What to do**:
  - **合并 Test A + Test C**：
    - Test A "scheduler switches at convergence point" 覆盖完整的阻塞-切换-汇聚生命周期
    - Test C "scheduler picks lowest non-blocked PC group after divergence" 覆盖最低 PC 选择策略
    - 合并为单一 TEST_CASE："scheduler switches at convergence and picks lowest non-blocked PC"
    - 保留 Test A 的完整 step_warp 序列（更完整），在关键步骤添加 Test C 的断言（如 `step_warp(w, v) == PATH_A_START` 后验证 `get_lanes_by_pc()` 大小）
  - **合并 Test D + Test E**：
    - Test D "boundary conv requires all active mask" 和 Test E "boundary non active mask no conv effect" 结构几乎相同
    - 合并为单一 TEST_CASE："boundary convergence requires all active_mask lanes"
    - 保留 Test D 的完整序列，在 Path A 到达 PC=14 后添加 Test E 的 `check_reconvergence() == false` 断言（验证非 active_mask 不影响收敛）
  - **保留 Test B** "two level div with convergence block"（独立价值，不合并）

  **Must NOT do**:
  - 不要丢失 Test C 的"最低 PC 选择"断言
  - 不要丢失 Test E 的"非 active_mask 不影响收敛"断言
  - 不要修改 `build_instrs()` 或 `setup()` helper

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: 需要仔细合并断言，确保不丢失覆盖

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 2,3,4,8 并行）
  - **Blocks**: Task 8（文档更新）
  - **Blocked By**: Task 1

  **Acceptance Criteria**:
  - [ ] 文件包含 3 个 TEST_CASE（原 5 个）
  - [ ] `ctest -R integration_divergence_sync_convergence -V` 通过
  - [ ] 合并后的 TEST_CASE 包含原 A/C 和 D/E 的所有关键断言

  **QA Scenarios**:
  ```
  Scenario: 验证合并后断言完整且测试通过
    Tool: Bash
    Steps:
      1. grep -c "TEST_CASE" tests/integration/divergence/test_divergence_sync_convergence.cpp
      2. cd build && cmake --build build
      3. ctest -R integration_divergence_sync_convergence -V 2>&1 | grep -E "passed|failed"
    Expected Result: (1) 3, (2) 0 errors, (3) 包含 "passed" 且不包含 "failed"
    Evidence: .omo/evidence/task-5-convergence-merge.log
  ```

  **Commit**: NO（合并到最终 commit）

---

- [x] 6. **清理 test_nested_divergence.cpp**

  **What to do**:
  - 删除 2 个元数据 TEST_CASE：
    - "test_nested_divergence: Structure verification"
    - "test_nested_divergence: Handler registration"
  - 删除 "test_nested_divergence: Register analysis"（测试 RegisterAnalyzer，与"嵌套分歧"无关）
  - 重命名保留的 TEST_CASE：
    - 原 "test_nested_divergence: Full warp execution with execute_warp_instruction"
    - 新名 "test_nested_predication: Full warp execution with nested setp+selp"
  - 添加文件头部注释：
    ```cpp
    /**
     * @brief 测试嵌套谓词化（setp + selp），非真实 SIMT 嵌套分歧
     * @note  本文件使用两级 setp/selp 谓词选择，不涉及 @%p bra 分支和 SIMT stack
     */
    ```
  - 添加 TODO 注释：
    ```cpp
    // TODO: 添加真正的嵌套分歧测试（两级 @%p bra 推送 SIMT stack entry）
    ```

  **Must NOT do**:
  - 不要添加真实的嵌套分歧测试（超出"仅重写断言"范围）
  - 不要修改 `build_nested_divergence_statements()` 的指令序列

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: 简单编辑和重命名

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 2,3,4,5 并行）
  - **Blocks**: Task 8（文档更新）
  - **Blocked By**: Task 1

  **Acceptance Criteria**:
  - [ ] 文件包含 1 个 TEST_CASE（原 4 个）
  - [ ] `ctest -R integration_nested_divergence -V` 通过
  - [ ] 文件头部有注释说明测试的是谓词化而非分歧
  - [ ] 保留的 TEST_CASE 名包含 "nested_predication"

  **QA Scenarios**:
  ```
  Scenario: 验证清理后测试通过且命名准确
    Tool: Bash
    Steps:
      1. grep -c "TEST_CASE" tests/integration/divergence/test_nested_divergence.cpp
      2. grep "nested_predication" tests/integration/divergence/test_nested_divergence.cpp
      3. cd build && cmake --build build && ctest -R integration_nested_divergence -V 2>&1 | grep "All tests passed"
    Expected Result: (1) 1, (2) 非空, (3) 包含 "All tests passed"
    Evidence: .omo/evidence/task-6-nested-cleanup.log
  ```

  **Commit**: NO（合并到最终 commit）

---

- [x] 7. **使用 `step_warp` 风格重写 standalone_integrated.cpp "all three modes" 测试**

  **What to do**:
  - 目标 TEST_CASE: "divergence_sync_standalone: all three modes produce same reconvergence"
  - 当前问题：循环跑 3 种模式，但断言完全相同（仅验证 PC=16/15/19），不验证模式差异；且使用 `execute_warp_instruction()` 直接驱动，未使用 `step_warp`
  - **重构方向：改用 `test_divergence_sync_convergence.cpp` 风格**：
    1. **指令构建**：使用 `ptxsim::testing` 工具重构指令序列：
       ```cpp
       using namespace ptxsim::testing;
       std::vector<StatementContext> v;
       v.push_back(make_mov("r_tid", "tid.x"));
       v.push_back(make_setp_lt("p_lt16", "r_tid", "16"));
       v.push_back(make_bra_pred("L_path_a", "p_lt16", false, /*reconv_pc=*/10));
       v.push_back(make_bra("L_path_b"));
       // ... L_path_a, L_path_b, L_join, bar.warp.sync, L_reduce, L_exit, ret
       ```
    2. **Predicate 设置**：使用 `setup_pred(w, 0x0000FFFFu)` 设置分歧 predicate
    3. **执行驱动**：使用 `step_warp(w, v)` 替代 `execute_warp_instruction()` 手动调用
    4. **断言强化**：
       - `REQUIRE(sm.get_divergence_execution_mode() == mode)` 验证模式设置
       - barrier 后 `CHECK(step_warp(w, v) == RECONV_PC)` 验证调度器选择
       - predicated branch 后验证 `get_exec_mask()` 反映分歧状态
       - 最终 `CHECK(warp->get_simt_stack().empty())` 验证汇聚完成
    5. 为每种模式记录 `INFO("Mode " << static_cast<int>(mode) << " completed")`
  - 当前断言数：约 12 个 → 目标：约 20 个

  **Must NOT do**:
  - 不要新增 TEST_CASE（在现有 "all three modes" TEST_CASE 内重构）
  - 不要添加需要新 StatementFactory 序列的场景
  - 不要改变测试的宏观意图（验证三种模式下屏障后分歧→汇聚行为一致）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: 现有代码熟悉后的断言增强

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 8 并行）
  - **Blocks**: F1-F4
  - **Blocked By**: Task 3（standalone 清理完成）

  **Acceptance Criteria**:
  - [ ] `ctest -R integration_divergence_sync_standalone -V` 通过
  - [ ] "all three modes" TEST_CASE 的断言数 ≥ 20（原 ~12）
  - [ ] 包含对 `get_divergence_execution_mode()`、`get_exec_mask()`、`get_simt_stack().empty()` 的断言

  **QA Scenarios**:
  ```
  Scenario: 验证强化后测试通过且断言增强
    Tool: Bash
    Steps:
      1. grep -c "CHECK\|REQUIRE" tests/integration/divergence/test_divergence_sync_standalone_integrated.cpp
      2. cd build && cmake --build build
      3. ctest -R integration_divergence_sync_standalone -V 2>&1 | grep "All tests passed"
    Expected Result: (1) ≥ 45（原 95 断言 - 删除 2 个元数据测试约 20 断言 + 新增 ~8 断言）, (2) 0 errors, (3) 包含 "All tests passed"
    Evidence: .omo/evidence/task-7-standalone-strengthen.log
  ```

  **Commit**: NO（合并到最终 commit）

---

- [x] 8. **更新所有外部文档引用**

  **What to do**:
  - **AGENTS.md**（根目录）行 207：
    - 移除 `integration_divergence_sync_isolated`
    - 确认 `integration_divergence_sync_convergence` 仍列在 divergence 标签下
    - 添加注释说明重构后的目标列表
  - **src/ptxsim/core/AGENTS.md** 行 49：
    - 更新注释为：
      ```
      // KNOWN ISSUE: synchronize_barrier() may not update active_mask correctly after barrier release
      // See: tests/integration/divergence/test_post_barrier_divergence.cpp (2 TEST_CASE documenting the issue)
      ```
  - **scripts/sanity.sh** 行 176：
    - 将 `test_divergence_sync_standalone|test_divergence_sync_isolated` 改为 `test_divergence_sync_standalone`
  - **docs/testing/TEST_DOCUMENTATION.md**：
    - 更新 `test_post_barrier_divergence.cpp` 条目：说明 5→2 TEST_CASE 合并，保留已知问题文档
  - **docs/adr/0013-statement-factory-test-unification.md**：
    - 更新表格：标记 `test_divergence_sync_isolated.cpp` 为"已删除（合并到 standalone）"
    - 标记 `test_post_barrier_divergence.cpp` 为"已合并（5→2 TEST_CASE）"
  - **workflow-state.md** 行 85：
    - 标记 `test_shortest_path_first.cpp` 为已删除
  - **.opencode/skills/ptx-lane-verification/SKILL.md**（仅记录，不强制更新）：
    - 添加注释说明 `test_divergence_sync_isolated.cpp` 和 `test_nested_divergence.cpp` 已被重构

  **Must NOT do**:
  - 不要修改 `docs/superpowers/plans/` 中的历史计划文件（它们记录历史状态）
  - 不要修改 `openspec/changes/*/tasks.md`（历史任务记录）
  - 不要修改 `docs/technical_design/implicit_reconvergence_enforcement.md`（除非其中的 Test A 引用需要更新名称）

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []
  - Reason: 文档更新

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 7 并行）
  - **Blocks**: F1-F4
  - **Blocked By**: Task 2,3,4,5,6（文件操作完成）

  **Acceptance Criteria**:
  - [ ] `grep -n "divergence_sync_isolated" AGENTS.md` 无输出
  - [ ] `grep -n "test_divergence_sync_isolated\|test_shortest_path_first" scripts/sanity.sh` 无输出（或仅存在于注释中）
  - [ ] `grep -n "5 TEST_CASE" docs/testing/TEST_DOCUMENTATION.md` 无输出（应更新为 2）

  **QA Scenarios**:
  ```
  Scenario: 验证所有外部引用已更新
    Tool: Bash
    Steps:
      1. grep -rn "integration_divergence_sync_isolated" AGENTS.md scripts/sanity.sh docs/ | grep -v "archive/" | grep -v "superpowers/plans/"
      2. grep -rn "test_shortest_path_first" AGENTS.md scripts/sanity.sh docs/ | grep -v "archive/" | grep -v "superpowers/plans/"
      3. grep -n "test_post_barrier_divergence" docs/testing/TEST_DOCUMENTATION.md
    Expected Result: (1) 空, (2) 空或仅在注释中, (3) 包含更新后的描述
    Evidence: .omo/evidence/task-8-docs-update.log
  ```

  **Commit**: NO（合并到最终 commit）

---

## Final Verification Wave

- [x] F1. **计划合规审计** — `oracle`
  读取计划端到端。对每个"Must Have"：验证实现存在（读文件、运行 ctest）。对每个"Must NOT Have"：搜索代码库中的禁止模式。检查证据文件存在于 `.omo/evidence/`。
  输出: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **代码质量审查** — `unspecified-high`
  运行 `cmake --build build` 检查编译警告。审查所有修改文件：无 `assert(false)`、无空 catch、无未使用变量。检查 AI slop：过度注释、过度抽象。
  输出: `Build [PASS/FAIL] | Warnings [N] | Files [N clean/N issues] | VERDICT`

- [x] F3. **回归测试执行** — `unspecified-high`
  从干净状态执行：`./scripts/sanity.sh` 完整运行。执行 `ctest -L "integration;divergence" -V`。对比基线断言数。
  输出: `Tests [N/N pass] | Assertions [baseline vs current] | Sanity [PASS/FAIL] | VERDICT`

- [x] F4. **范围保真检查** — `deep`
  对每个任务：读取"What to do"，读取实际 diff。验证 1:1 — 规格中要求做的都做了，规格外没做。检查"Must NOT do"合规性。
  输出: `Tasks [N/N compliant] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

采用单 commit 完成整个重构（原子性保证）：

```
refactor(tests): consolidate divergence integration tests

- Delete test_shortest_path_first.cpp (no real ShortestFirst behavior tested)
- Delete test_divergence_sync_isolated.cpp (duplicate of standalone)
- Merge post_barrier tests: 5→2 TEST_CASE (retain known-issue docs)
- Merge convergence tests: A+C, D+E
- Remove metadata tests (structure/handler registration) from 3 files
- Strengthen "all three modes" assertions in standalone
- Update CMakeLists.txt, sanity.sh, AGENTS.md

Refs: AGENTS.md#48 (known issue: synchronize_barrier active_mask)
```

**Pre-commit verification**:
```bash
. env.sh
cmake --build build
ctest -L "integration;divergence" -V
./scripts/sanity.sh
```

---

## Success Criteria

### Verification Commands
```bash
# 1. 编译
. env.sh && cmake --build build  # Expected: 0 errors, 0 warnings

# 2. Divergence 测试全部通过
cd build && ctest -L "integration;divergence" -V  # Expected: 4/4 files pass, ~11 TEST_CASE

# 3. Sanity 零回归
./scripts/sanity.sh  # Expected: All tests passed!

# 4. 断言数检查（允许减少但不过度）
# 基线: ~207 assertions / 28 TEST_CASE / 6 files（以 Task 1 实际输出为准）
# 目标: ≥ 130 assertions / ~11 TEST_CASE / 4 文件
# 预估: 207 - 删除 shortest (~9) - 删除 iso (~20) - 合并收敛 (~15) - 清理元数据 (~25) - 合并 post_barrier (~6) + 强化 (~8) = ~140
```

### Final Checklist
- [x] 所有 "Must Have" 已满足
- [x] 所有 "Must NOT Have" 未违反
- [x] 编译零错误零警告
- [x] `ctest -L "integration;divergence"` 100% 通过
- [x] `./scripts/sanity.sh` 零回归
- [x] 外部文档引用已更新（AGENTS.md、sanity.sh、skills）
- [x] 已知问题文档保留（post_barrier active_mask bug）
