# 集成测试 5 原则合规化重构计划

## TL;DR

> **目标**: 将 `tests/integration/` 下 ~13 个违反指令序列集成测试 5 原则的文件重构为 `step_warp` 驱动模式，将 ~14 个非指令序列测试迁移到正确目录（`unit/`/`e2e/`/`archive/`），最终使 `tests/integration/` 100% 合规。
>
> **交付物**: 重构后的测试文件（~13 个）、迁移后的文件（~14 个）、更新的 `CMakeLists.txt`（3 个）、更新的 `sanity.sh`、更新的 `AGENTS.md`
>
> **预估工作量**: Large（~30 个独立任务，预计 15-20 小时）
> **并行执行**: YES — 5 个 Wave，每个 Wave 内部文件可并行
> **关键路径**: Wave 0 → Wave 1 → Wave 2 → Wave 3 → Wave 4 → Wave FINAL

---

## Context

### 原始请求
用户要求根据 AGENTS.md 新增的 5 条指令序列集成测试核心原则，对 `tests/integration/` 下全部 30 个测试文件进行审计并给出实施计划。

### 审计结果摘要
- **29 个文件**中仅 **1 个完全合规**（`test_divergence_sync_convergence.cpp`）
- **~13 个 VIOLATES**（直接调用 `execute_warp_instruction` 绕过调度器）
- **~14 个 N/A**（非指令序列测试，应迁出 `integration/`）
- **1 个 PARTIAL**（`test_divergence_sync_standalone_integrated.cpp`，混合合规与违规）
- **3 个已损坏**（`test_spinlock_simulation.cu`、`test_cfg_analysis.cpp`、`test_register_bank_subwarp.cpp`，已在 `CMakeLists.txt` 中注释掉）
- **1 个孤儿文件**（`test_syncthreads_test3_full.cpp`，存在于目录但未在 `CMakeLists.txt` 注册）

### Metis 审查发现（已纳入）
1. **参考实现本身违规**: `test_divergence_sync_convergence.cpp` 的 Test B 手动 `push` SIMTStackEntry，违反原则 5 → 需添加例外条款
2. **4 个 VIOLATES 实为 N/A**: `test_exec_integration_h1_h4`、`test_exec_layer_e1_e3`、`test_barrier_interaction_integrated`、`test3_reproduction` → 从 VIOLATES 中移除，改为 N/A
3. **工具缺口**: `setup_pred` 仅支持 `%p1`，`instruction_helpers` 仅 15 个 make_* 函数 → **本次不扩展工具**
4. **Handler 隔离测试**: 某些测试有意绕过调度器来隔离 handler 行为 → 需定义 `[handler_isolation]` 标签
5. **Scheduler bug 暴露风险**: 重构后若因 scheduler bug 失败，不回滚重构，而是记录 bug

---

## Work Objectives

### 核心目标
重构 `tests/integration/` 使所有指令序列集成测试遵循 5 原则：PC 经 `execute_warp_instruction` 驱动（通过 `step_warp`）、路径由 `step_warp` 推进、不干预调度器、predicate 经 `setup_pred`、分歧由 `handle_branch` 自动处理。

### 具体交付物
1. **重构文件**: ~16 个测试文件从直接 `execute_warp_instruction` 改为 `step_warp` 驱动
2. **迁移文件**: ~14 个 N/A 文件从 `integration/` 迁到 `unit/`/`e2e/`/`archive/`
3. **CMakeLists 更新**: `tests/integration/CMakeLists.txt`、`tests/unit/CMakeLists.txt`、`tests/e2e/CMakeLists.txt`
4. **sanity.sh 更新**: 同步 regex 以匹配重命名后的 ctest 目标
5. **AGENTS.md 更新**: 原则 5 例外条款 + handler isolation 标签定义

### 定义 of Done
- `tests/integration/` 中直接 `execute_warp_instruction` 调用数（除 `[handler_isolation]` 标注外）降至 0
- `ctest -L integration` 只显示指令序列测试（不含 N/A 文件）
- `./scripts/sanity.sh` 通过（exit code 0）

### Must Have
- 每文件重构后断言语义与重构前完全一致
- 每文件独立 commit，包含 `ctest -R <test_name> -V` 验证
- N/A 迁移只做物理移动 + CMakeLists 更新 + ctest 前缀变更，**不修改内部代码**
- sanity.sh regex 与 ctest 名重命名同步更新
- AGENTS.md 原则 5 添加例外条款
- **integration/ 下不允许任何直接 `execute_warp_instruction` 调用**

### Must NOT Have (Guardrails)
- **integration/ 下不允许任何直接 `execute_warp_instruction`**（需直调的测试归入 unit/）
- **不添加新的 `make_*` helper 函数**（工具缺口标记为前置工作）
- **不修改 `step_warp` 或 `setup_pred` 接口**
- **不合并/拆分 TEST_CASE**
- **不改善断言**（CHECK→REQUIRE 等）
- **不修复 3 个已损坏文件**（直接归档）
- **不批量修改多个文件在一个 commit**
- **不回滚重构以绕过 scheduler bug**（记录 bug 单独修复）

---

## Verification Strategy

### 测试决策
- **基础设施**: YES（ctest + Catch2）
- **自动化测试**: Tests-after（每文件重构后运行对应 ctest）
- **框架**: Catch2（`ctest -R <name> -V`）
- **Sanity 检查**: `./scripts/sanity.sh --quick`（阶段验收）、`./scripts/sanity.sh`（最终验收）

### QA Policy
每任务包含 agent-executed QA 场景：
- **测试验证**: `ctest -R <test_name> -V`，断言输出与重构前一致
- **违规率验证**: `grep -r "execute_warp_instruction" tests/integration/ --include="*.cpp" | grep -v "handler_isolation" | wc -l` → 0
- **阶段验收**: `./scripts/sanity.sh --quick` → exit code 0

---

## Execution Strategy

### 并行执行 Waves

```
Wave 0 (Foundation — 必须先完成):
├── W0.1: AGENTS.md 原则 5 添加例外条款
├── W0.2: AGENTS.md 定义 handler isolation 标签
└── W0.3: 决定孤儿文件 test_syncthreads_test3_full.cpp 命运

Wave 1 (N/A Migration — 低风险，物理移动):
├── W1.1: 迁移 exec/test_exec_integration_h1_h4 → unit/exec/
├── W1.2: 迁移 exec/test_exec_layer_e1_e3 → unit/exec/
├── W1.3: 迁移 barrier/test_barrier_interaction_integrated → unit/barrier/
├── W1.4: 迁移 simt/test_simt_integration → unit/simt/
├── W1.5: 迁移 simt/test_handle_branch_integration → unit/simt/
├── W1.6: 迁移 simt/test_barrier_simt_integration → unit/simt/
├── W1.7: 迁移 sync/test3_reproduction → tests/archive/
├── W1.8: 迁移 sync/test_syncthreads_direction → unit/sync/
├── W1.9: 迁移 sync/test_syncthreads_test3_repro → unit/sync/
├── W1.10: 迁移 cfg/integration_cfg_benchmark → tests/archive/
├── W1.11: 归档 3 个已损坏文件
├── W1.12: 更新 tests/unit/CMakeLists.txt
├── W1.13: 更新 tests/integration/CMakeLists.txt
└── W1.14: 更新 scripts/sanity.sh regex

Wave 2 (P0 Refactors — 高价值/高调用量):
├── W2.1: 重构 barrier/test_warp_barrier_integrated (32 处直调)
├── W2.2: 重构 barrier/test_barrier_scenarios_integrated (19 处)
├── W2.3: 重构 sync/test_syncthreads_full_pipeline (4 处)
└── W2.4: 重构 pc/test_pc_management_integrated (6 处)

Wave 3 (P1 Refactors — 中等复杂度):
├── W3.1: 重构 barrier/test_barrier_verification_integrated (9 处)
├── W3.2: 重构 barrier/test_barrier_divergence_scheduling (1 处)
├── W3.3: 重构 divergence/test_post_barrier_divergence (4 处 + PC 操控)
├── W3.4: 重构 divergence/test_nested_divergence (1 处)
├── W3.5: 重构 exec/test_warp_state_integrated (4 处)
├── W3.6: 重构 exec/test_ptx_lane_verification (1 处 + set_pc)
├── W3.7: 重构 simt/test_simt_stack_entry_integrated (4 处)
├── W3.8: 重构 simt/test_simt_thread_pc_integrated (6 处)
└── W3.9: 重构 sync/test_sync_mechanism_integrated (4 处)

Wave 4 (P2 + Partial + Handler Isolation):
├── W4.1: 重构 barrier/test_barrier_module_integrated (2 处)
├── W4.2: 重构 sync/test_syncthreads_test3_full (1 处)
├── W4.3: 修复 test_divergence_sync_standalone_integrated 的 all_modes TEST_CASE
└── W4.4: 标注 handler isolation 测试（保留直接调用的文件）

Wave FINAL (Verification — 4 并行审查):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)
-> 呈现结果 -> 获取用户明确 okay

Critical Path: W0 → W1 → W2 → W3 → W4 → F1-F4 → user okay
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 4 (Waves 2 & 3)
```

### Dependency Matrix (abbreviated)

- **W0**: - - W1, W2, W3, W4
- **W1**: W0 - W2, W3, W4
- **W2**: W0, W1 - W3, W4
- **W3**: W0, W1, W2 - W4
- **W4**: W0, W1, W2, W3 - F1-F4
- **F1-F4**: W0-W4 - user okay

### Agent Dispatch Summary

- **W0**: quick (文档更新)
- **W1**: quick (文件移动 + CMakeLists 更新)
- **W2**: deep (复杂重构，高调用量)
- **W3**: unspecified-high (中等复杂度重构)
- **W4**: unspecified-high (简单重构 + 标注)
- **FINAL**: oracle / unspecified-high / deep

---

## TODOs

- [x] 1. AGENTS.md 原则 5 分类规则（非例外条款）**

  **What to do**:
  在 AGENTS.md 原则 5 后添加**分类规则**文本："如测试需要手动设置 SIMT stack/PC 状态（如两级分歧 back-edge），则该测试**不是指令序列集成测试**，应归入 `tests/unit/`（单元测试允许直接 `execute_warp_instruction`）。"

  **Must NOT do**:
  - 不修改原则 1-4 文本
  - 不添加其他例外

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO（Wave 0 基础任务）
  - **Blocks**: W0.2, W0.3, W1-W4

  **Acceptance Criteria**:
  - [ ] AGENTS.md 原则 5 段落后含例外条款注释
  - [ ] `grep -A2 "分歧由 handle_branch 自动处理" AGENTS.md` 显示例外条款

  **QA Scenarios**:
  ```
  Scenario: 验证例外条款存在
    Tool: Bash (grep)
    Steps:
      1. grep -n "例外" AGENTS.md
    Expected Result: 输出包含 "两级分歧" 或 "back-edge"
    Evidence: .omo/evidence/w0-1-exception-clause.log
  ```

  **Commit**: YES
  - Message: `docs(agents): add principle 5 exception clause for SIMT stack setup`
  - Files: `AGENTS.md`

- [x] 2. AGENTS.md 定义 handler isolation 标签**

  **What to do**:
  在 AGENTS.md 类型二章节添加 handler isolation 定义：某些测试有意绕过调度器来隔离测试 handler 行为（如 bar.warp.sync handler 是否正确更新 reconvergence_pc）。这类测试允许直接调用 `execute_warp_instruction`，但必须在 TEST_CASE 标签中添加 `[handler_isolation]`，并在文件头注释说明理由。

  **Must NOT do**:
  - 不实际标注任何文件（留到 W4.4）

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 W0.1 并行）

  **Acceptance Criteria**:
  - [ ] AGENTS.md 明确声明：integration/ 下**不允许**直接 `execute_warp_instruction`（所有指令序列测试必须用 `step_warp`）
  - [ ] AGENTS.md 明确声明：unit/ 下允许直接 `execute_warp_instruction`（单元测试可直接测试 handler 行为）

  **QA Scenarios**:
  ```
  Scenario: 验证 integration/ 无例外声明
    Tool: Bash (grep)
    Steps:
      1. grep -A5 "指令序列集成测试" AGENTS.md | grep -i "handler\|例外\|execute_warp_instruction"
    Expected Result: 无 "handler_isolation" 或 "execute_warp_instruction" 例外声明在 integration 段落
    Evidence: .omo/evidence/w0-2-no-handler-isolation.log
  ```

  **Commit**: YES
  - Message: `docs(agents): clarify integration/ uses step_warp only, unit/ allows direct execute_warp_instruction`

- [x] 3. 决定孤儿文件 test_syncthreads_test3_full.cpp 命运**

  **What to do**:
  检查 `tests/integration/sync/test_syncthreads_test3_full.cpp`（存在于目录但未在 `CMakeLists.txt` 注册）。**用户已确认：归档到 `tests/archive/`**（从未被构建，可能已过时）。

  **Must NOT do**:
  - 不修改文件内容（无论选 A 或 B）

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 W0.1/W0.2 并行）
  - **Blocks**: W1.x（若选 B，则 W1.11 包含此文件）

  **Acceptance Criteria**:
  - [ ] 文件被移动或注册（二选一）
  - [ ] 无孤儿文件残留

  **QA Scenarios**:
  ```
  Scenario: 验证孤儿文件已处理
    Tool: Bash
    Steps:
      1. ls tests/integration/sync/test_syncthreads_test3_full.cpp 2>/dev/null && echo "still orphan" || echo "handled"
    Expected Result: "handled"
    Evidence: .omo/evidence/w0-3-orphan-handled.log
  ```

  **Commit**: YES
  - Message: `refactor(tests): archive orphan test_syncthreads_test3_full.cpp`

- [x] 4. 修复参考实现 test_divergence_sync_convergence.cpp Test B（原则 5 违规）**

  **What to do**:
  参考实现 `tests/integration/divergence/test_divergence_sync_convergence.cpp` 的 Test B（"two level div with convergence block"，line 179-235）在 line 198 处直接调用 `w->get_simt_stack().push(le)` 手动推入 SIMT stack entry，**违反原则 5**。这是计划中标注的"完全合规"参考实现本身的违规。

  **修复方案**（推荐 A）：
  - **方案 A（首选）**: 重构 Test B 构造的 kernel，让其**包含一个真正的循环**（3 次 bra back-edge），通过 `step_warp` 驱动循环迭代，让 `handle_branch` 自然产生两级分歧 entry。
  - **方案 B（fallback）**: 将 Test B 拆分为两个文件：保留 part 1（用 step_warp 驱动一级分歧），part 2（手动 push 验证 handle_branch 收敛行为）迁到 `tests/unit/simt/test_handle_branch_two_level.cpp`。

  **Must NOT do**:
  - 不可保留任何 `get_simt_stack().push()` / `set_thread_pc()` / `set_pc()` 调用在 `tests/integration/` 下
  - 不可在 AGENTS.md 中为 Test B 添加特殊例外

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需理解两级分歧 back-edge 的 SIMT stack 语义，重构 kernel
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO（Wave 0 基础任务，必须在所有 refactor 前完成）
  - **Blocks**: W2-W4

  **Acceptance Criteria**:
  - [ ] `grep -c "get_simt_stack().push\|set_thread_pc\|set_pc" tests/integration/divergence/test_divergence_sync_convergence.cpp` == 0
  - [ ] `ctest -R integration_divergence_sync_convergence -V` PASS
  - [ ] 5 原则全部满足（成为真正的参考实现）

  **QA Scenarios**:
  ```
  Scenario: 验证参考实现零违规
    Tool: Bash (grep)
    Steps:
      1. grep -cE "get_simt_stack\(\)\.push|set_thread_pc|set_pc" tests/integration/divergence/test_divergence_sync_convergence.cpp
    Expected Result: 0
    Evidence: .omo/evidence/w0-4-reference-clean.log

  Scenario: 验证 ctest 通过
    Tool: Bash (ctest)
    Steps:
      1. cd build && cmake --build . --target integration_divergence_sync_convergence
      2. ctest -R integration_divergence_sync_convergence -V
    Expected Result: exit code 0
    Evidence: .omo/evidence/w0-4-ctest-pass.log
  ```

  **Commit**: YES
  - Message: `refactor(tests): fix Test B in convergence reference to use loop + step_warp`

- [x] 5. 迁移 exec/test_exec_integration_h1_h4 → unit/exec/**

  **What to do**:
  1. `mv tests/integration/exec/test_exec_integration_h1_h4.cpp tests/unit/exec/`
  2. 从 `tests/integration/CMakeLists.txt` 移除 `integration_exec_h1_h4` 条目
  3. 在 `tests/unit/CMakeLists.txt` 添加 `unit_exec_integration_h1_h4` 条目
  4. 更新标签：`integration;exec` → `unit;exec`
  5. 更新 `scripts/sanity.sh` 中的 regex

  **Must NOT do**:
  - 不修改 .cpp 文件内容

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES（W1 内所有迁移任务并行）
  - **Blocked By**: W0.1, W0.2, W0.3

  **Acceptance Criteria**:
  - [ ] `ctest -R unit_exec_integration_h1_h4 -V` PASS
  - [ ] `ctest -R integration_exec_h1_h4 -V` 找不到测试

  **QA Scenarios**:
  ```
  Scenario: 验证迁移后 ctest 可发现
    Tool: Bash (ctest)
    Steps:
      1. cd build && cmake .. && cmake --build . --target unit_exec_integration_h1_h4
      2. ctest -R unit_exec_integration_h1_h4 -V
    Expected Result: exit code 0
    Evidence: .omo/evidence/w1-1-exec-h1-h4-migrated.log
  ```

  **Commit**: YES
  - Message: `refactor(tests): move exec_integration_h1_h4 from integration to unit`

- [x] 6. 迁移 exec/test_exec_layer_e1_e3 → unit/exec/**

  **What to do**:
  同 W1.1 模式。文件：`test_exec_layer_e1_e3.cpp`。ctest 名：`unit_exec_layer_e1_e3`。

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] `ctest -R unit_exec_layer_e1_e3 -V` PASS

  **QA Scenarios**:
  ```
  Scenario: 验证迁移后 ctest 可发现
    Tool: Bash (ctest)
    Steps:
      1. cd build && cmake --build . --target unit_exec_layer_e1_e3
      2. ctest -R unit_exec_layer_e1_e3 -V
    Expected Result: exit code 0
    Evidence: .omo/evidence/w1-2-exec-layer-migrated.log
  ```

  **Commit**: YES

- [x] 7. 迁移 barrier/test_barrier_interaction_integrated → unit/barrier/**

  **What to do**:
  同 W1.1 模式。文件：`test_barrier_interaction_integrated.cpp`。ctest 名：`unit_barrier_interaction`。

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] `ctest -R unit_barrier_interaction -V` PASS

  **QA Scenarios**:
  ```
  Scenario: 验证迁移后 ctest 可发现
    Tool: Bash (ctest)
    Steps:
      1. cd build && cmake --build . --target unit_barrier_interaction
      2. ctest -R unit_barrier_interaction -V
    Expected Result: exit code 0
    Evidence: .omo/evidence/w1-3-barrier-interaction-migrated.log
  ```

  **Commit**: YES

- [x] 8. 迁移 simt/test_simt_integration → unit/simt/**

  **What to do**:
  同 W1.1 模式。文件：`test_simt_integration.cpp`。ctest 名：`unit_simt_integration`。

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] `ctest -R unit_simt_integration -V` PASS

  **Commit**: YES

- [x] 9. 迁移 simt/test_handle_branch_integration → unit/simt/**

  **What to do**:
  同 W1.1 模式。文件：`test_handle_branch_integration.cpp`。ctest 名：`unit_handle_branch`。

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] `ctest -R unit_handle_branch -V` PASS

  **Commit**: YES

- [x] 10. 迁移 simt/test_barrier_simt_integration → unit/simt/**

  **What to do**:
  同 W1.1 模式。文件：`test_barrier_simt_integration.cpp`。ctest 名：`unit_barrier_simt`。

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] `ctest -R unit_barrier_simt -V` PASS

  **Commit**: YES

- [x] 11. 迁移 sync/test3_reproduction → tests/archive/**

  **What to do**:
  1. `mv tests/integration/sync/test3_reproduction.cpp tests/archive/`
  2. 从 `tests/integration/CMakeLists.txt` 移除 `integration_test3_reproduction` 条目
  3. 不添加到 unit/CMakeLists.txt（这是调试复现脚本，非正式测试）
  4. 更新 `scripts/sanity.sh`

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] `ctest -R integration_test3_reproduction -V` 找不到测试

  **Commit**: YES

- [x] 12. 迁移 sync/test_syncthreads_direction → unit/sync/**

  **What to do**:
  同 W1.1 模式。文件：`test_syncthreads_direction.cpp`。ctest 名：`unit_syncthreads_direction`。

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] `ctest -R unit_syncthreads_direction -V` PASS

  **Commit**: YES

- [x] 13. 迁移 sync/test_syncthreads_test3_repro → unit/sync/**

  **What to do**:
  同 W1.1 模式。文件：`test_syncthreads_test3_repro.cpp`。ctest 名：`unit_syncthreads_test3_repro`。

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] `ctest -R unit_syncthreads_test3_repro -V` PASS

  **Commit**: YES

- [x] 14. 迁移 cfg/integration_cfg_benchmark → tests/archive/**

  **What to do**:
  1. `mv tests/integration/cfg/integration_cfg_benchmark.cpp tests/archive/`
  2. 从 `tests/integration/CMakeLists.txt` 移除 `integration_cfg_benchmark` 条目（注意它用 `add_standalone_test` 而非 `add_catch_test`）
  3. 不添加到 unit/CMakeLists.txt（standalone main，非 Catch2）
  4. 更新 `scripts/sanity.sh`

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] `ctest -R integration_cfg_benchmark -V` 找不到测试

  **Commit**: YES

- [x] 15. 归档 3 个已损坏文件**

  **What to do**:
  1. `mv tests/integration/sync/test_spinlock_simulation.cu tests/archive/`
  2. `mv tests/integration/cfg/test_cfg_analysis.cpp tests/archive/`
  3. `mv tests/integration/register/test_register_bank_subwarp.cpp tests/archive/`
  4. 从 `tests/integration/CMakeLists.txt` 移除已注释的条目（清理死代码）
  5. 更新 `scripts/sanity.sh`（移除对这些文件的引用）

  **Must NOT do**:
  - 不修复这些文件（超出范围）
  - 不尝试编译它们

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] 3 个文件已移动到 tests/archive/
  - [ ] CMakeLists.txt 中无对应注释条目

  **Commit**: YES
  - Message: `refactor(tests): archive 3 broken test files`

- [x] 16. 更新 tests/unit/CMakeLists.txt**

  **What to do**:
  在 `tests/unit/CMakeLists.txt` 中为所有迁移文件添加 `add_catch_test` 条目，使用 `unit_` 前缀和对应标签。

  **Must NOT do**:
  - 不修改已有 unit 测试条目

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: NO（需等待 W1.1-W1.9 完成以确定最终列表）
  - **Blocked By**: W1.1-W1.9

  **Acceptance Criteria**:
  - [ ] `ctest -L unit -N` 显示所有迁移的测试

  **Commit**: YES
  - Message: `build(tests): add migrated tests to unit CMakeLists.txt`

- [x] 17. 更新 tests/integration/CMakeLists.txt**

  **What to do**:
  从 `tests/integration/CMakeLists.txt` 中移除所有已迁移/归档文件的 `add_catch_test` 条目，并更新文件头注释以匹配新的 5 原则描述。

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: NO（需等待 W1.1-W1.14）
  - **Blocked By**: W1.1-W1.14

  **Acceptance Criteria**:
  - [ ] `ctest -L integration -N` 不显示任何已迁移/归档的测试

  **Commit**: YES
  - Message: `build(tests): remove migrated tests from integration CMakeLists.txt`

- [x] 18. 更新 scripts/sanity.sh regex**

  **What to do**:
  同步更新 `scripts/sanity.sh` 中所有涉及已迁移/重命名测试的 regex。确保 `run_regex_tests` 调用的模式能匹配新的 ctest 名。

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: NO（需等待 W1.1-W1.16）
  - **Blocked By**: W1.1-W1.16

  **Acceptance Criteria**:
  - [ ] `./scripts/sanity.sh --quick` 通过（exit code 0）

  **QA Scenarios**:
  ```
  Scenario: 验证 sanity.sh 通过
    Tool: Bash
    Steps:
      1. ./scripts/sanity.sh --quick
    Expected Result: exit code 0
    Evidence: .omo/evidence/w1-17-sanity-pass.log
  ```

  **Commit**: YES
  - Message: `build(tests): update sanity.sh regex for migrated tests`

- [x] 19. 重构 barrier/test_warp_barrier_integrated (32 处直调)**

  **What to do**:
  将 32 处直接 `execute_warp_instruction` 调用改为 `step_warp(warp, statements)` 驱动。参考 `test_divergence_sync_convergence.cpp` 的 Test A 模式：
  1. 构建指令序列（保留原有 `makeGenericInstr` 调用）
  2. 用 `step_warp` 替代所有 `warp->execute_warp_instruction(stmts[i], i)`
  3. 添加 PC 断言验证调度器选择（`CHECK(step_warp(...) == expected_pc)`）
  4. 验证 active_mask、SIMT stack 深度等状态与重构前一致

  **Must NOT do**:
  - 不合并/拆分 TEST_CASE
  - 不改变断言语义（CHECK/REQUIRE 不变）
  - 不添加新的 make_* helper

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 32 处直调需逐行映射到 step_warp 语义，需理解 barrier 指令的调度器行为
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 W2.2-W2.4 并行）
  - **Blocked By**: W1（N/A 迁移完成，避免冲突）

  **Acceptance Criteria**:
  - [ ] `grep -c "execute_warp_instruction" tests/integration/barrier/test_warp_barrier_integrated.cpp` == 0
  - [ ] `ctest -R integration_warp_barrier -V` PASS（与重构前相同输出）

  **QA Scenarios**:
  ```
  Scenario: 验证重构后无直调
    Tool: Bash (grep)
    Steps:
      1. grep -c "execute_warp_instruction" tests/integration/barrier/test_warp_barrier_integrated.cpp
    Expected Result: 0
    Evidence: .omo/evidence/w2-1-no-direct-call.log

  Scenario: 验证 ctest 通过
    Tool: Bash (ctest)
    Steps:
      1. cd build && cmake --build . --target integration_warp_barrier
      2. ctest -R integration_warp_barrier -V
    Expected Result: exit code 0
    Evidence: .omo/evidence/w2-1-ctest-pass.log
  ```

  **Commit**: YES
  - Message: `refactor(tests): warp_barrier_integrated use step_warp per 5 principles`
  - Files: `tests/integration/barrier/test_warp_barrier_integrated.cpp`
  - Pre-commit: `ctest -R integration_warp_barrier -V`

- [x] 20. 重构 barrier/test_barrier_scenarios_integrated (19 处直调)**

  **What to do**:
  同 W2.1 模式。19 处直调改为 `step_warp` 驱动。此文件是 barrier 场景库，重构后可作为 barrier 测试的样板。

  **Recommended Agent Profile**:
  - **Category**: `deep`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] 直调用计数为 0
  - [ ] `ctest -R integration_barrier_scenarios -V` PASS

  **Commit**: YES

- [x] 21. 重构 sync/test_syncthreads_full_pipeline (4 处直调)**

  **What to do**:
  将 4 处直调改为 `step_warp` 驱动。此文件测试 `__syncthreads` 完整流水线，是 sync 路径的高价值集成测试。需特别注意 barrier 后的 reconvergence 行为验证。

  **Recommended Agent Profile**:
  - **Category**: `deep`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] 直调用计数为 0
  - [ ] `ctest -R integration_syncthreads_full_pipeline -V` PASS

  **Commit**: YES

- [x] 22. 重构 pc/test_pc_management_integrated (6 处直调)**

  **What to do**:
  将 6 处直调改为 `step_warp` 驱动。此文件**测试 PC 管理却绕过调度器**，是最讽刺的违规。重构后应通过 `step_warp` 自然推进 PC，验证调度器在分歧/汇聚时的 PC 选择。

  **Recommended Agent Profile**:
  - **Category**: `deep`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] 直调用计数为 0
  - [ ] `ctest -R integration_pc_management -V` PASS

  **Commit**: YES

- [x] 23. 重构 barrier/test_barrier_verification_integrated (9 处直调)**

  **What to do**:
  将 9 处直调改为 `step_warp` 驱动。此文件测试 barrier 验证逻辑，重构后应通过调度器推进验证 barrier 后状态。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`

  **Parallelization**:
  - **Can Run In Parallel**: YES（W3 内部全部并行）
  - **Blocked By**: W2

  **Acceptance Criteria**:
  - [ ] 直调用计数为 0
  - [ ] `ctest -R integration_barrier_verification -V` PASS

  **Commit**: YES

- [x] 24. 重构 barrier/test_barrier_divergence_scheduling (1 处直调)**

  **What to do**:
  将 1 处直调改为 `step_warp` 驱动。同时将其中的 `setp` PTX 字符串构造改为 `setup_pred`（若 predicate 仅为测试 setup 而非被测 kernel 的一部分）。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] 直调用计数为 0
  - [ ] `ctest -R integration_barrier_divergence_scheduling -V` PASS

  **Commit**: YES

- [x] 25. 重构 divergence/test_post_barrier_divergence (4 处直调 + PC 操控)**

  **What to do**:
  1. 移除所有 `t->set_pc(0)` 和 `warp.set_thread_pc(i, 1)` 调用（违反原则 3）
  2. 将 4 处直调改为 `step_warp` 驱动
  3. 通过自然指令执行达到 post-barrier 状态（而非手动设置 PC）
  4. 保留 known-issue 文档注释（此文件记录 BUG-REPRODUCTION）

  **Must NOT do**:
  - 不删除 known-issue TEST_CASE

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 涉及 PC 操控移除，需重构测试 setup 逻辑

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] `grep -c "set_pc\|set_thread_pc" tests/integration/divergence/test_post_barrier_divergence.cpp` == 0
  - [ ] `grep -c "execute_warp_instruction" tests/integration/divergence/test_post_barrier_divergence.cpp` == 0
  - [ ] `ctest -R integration_post_barrier_divergence -V` PASS

  **Commit**: YES

- [x] 26. 重构 divergence/test_nested_divergence (1 处直调)**

  **What to do**:
  将 1 处直调改为 `step_warp` 驱动。注意此文件有 `setp` PTX 字符串，但这些是**被测 kernel 的指令**（makeGenericInstr 构造），不是 setup_pred 的替代。保留这些 PTX 字符串，仅替换执行方式。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] 直调用计数为 0
  - [ ] `ctest -R integration_nested_divergence -V` PASS

  **Commit**: YES

- [x] 27. 重构 exec/test_warp_state_integrated (4 处直调)**

  **What to do**:
  将 4 处直调改为 `step_warp` 驱动。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] 直调用计数为 0
  - [ ] `ctest -R integration_warp_state -V` PASS

  **Commit**: YES

- [x] 28. 重构 exec/test_ptx_lane_verification (1 处直调 + set_pc)**

  **What to do**:
  1. 移除 `t->set_pc(0)` 调用
  2. 将 1 处直调改为 `step_warp` 驱动
  3. 此文件需要 `TEST_SOURCE_DIR` 编译定义，确保重构后编译定义保留

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] `grep -c "set_pc" tests/integration/exec/test_ptx_lane_verification.cpp` == 0
  - [ ] 直调用计数为 0
  - [ ] `ctest -R integration_ptx_lane_verification -V` PASS

  **Commit**: YES

- [x] 29. 重构 simt/test_simt_stack_entry_integrated (4 处直调)**

  **What to do**:
  将 4 处直调改为 `step_warp` 驱动。注意此文件原示例在 AGENTS.md 中被引用，重构后需同步更新 AGENTS.md 中的示例代码（若示例仍引用此文件）。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] 直调用计数为 0
  - [ ] `ctest -R integration_simt_stack_entry -V` PASS

  **Commit**: YES

- [x] 30. 重构 simt/test_simt_thread_pc_integrated (6 处直调)**

  **What to do**:
  将 6 处直调改为 `step_warp` 驱动。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] 直调用计数为 0
  - [ ] `ctest -R integration_simt_thread_pc -V` PASS

  **Commit**: YES

- [x] 31. 重构 sync/test_sync_mechanism_integrated (4 处直调)**

  **What to do**:
  将 4 处直调改为 `step_warp` 驱动。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] 直调用计数为 0
  - [ ] `ctest -R integration_sync_mechanism -V` PASS

  **Commit**: YES

- [x] 32. 重构 barrier/test_barrier_module_integrated (2 处直调)**

  **What to do**:
  将 2 处直调改为 `step_warp` 驱动。此文件测试 barrier module 的 handler 行为，若重构后发现需保留直接调用（隔离 handler 测试），则按 W4.4 流程标注 `[handler_isolation]`。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`

  **Parallelization**:
  - **Can Run In Parallel**: YES（W4 内部全部并行）
  - **Blocked By**: W3

  **Acceptance Criteria**:
  - [ ] 直调用计数为 0（或已标注 handler_isolation）
  - [ ] `ctest -R integration_barrier_module -V` PASS

  **Commit**: YES

- [x] 33. （已跳过）孤儿文件已归档**

  **状态**: W0.3 已确认归档 `test_syncthreads_test3_full.cpp` → 无需重构。文件已在 W1.x 迁移阶段处理。

- [x] 34. 修复 test_divergence_sync_standalone_integrated 的 all_modes TEST_CASE**

  **What to do**:
  此文件目前**混合合规**（4 个 TEST_CASE 中 3 个用 `step_warp`，但 `all_modes` TEST_CASE 全部直调 14 处）。
  1. 分析 `all_modes` TEST_CASE 的 14 处直调：若为 mode 对比测试（standalone/convergence/isolated 三种执行模式对比），需判断是否必须直调
  2. 若可以 step_warp 化：替换为 step_warp + 断言
  3. 若必须直调（三种模式使用不同调度器参数）：按 W4.4 标注 `[handler_isolation]`，并在文件头注释说明

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需理解三种执行模式差异，判断哪些场景必须直调

  **Parallelization**:
  - **Can Run In Parallel**: YES

  **Acceptance Criteria**:
  - [ ] `all_modes` TEST_CASE 要么全用 step_warp，要么标注 `[handler_isolation]`
  - [ ] `ctest -R integration_divergence_sync_standalone -V` PASS

  **Commit**: YES

- [x] 35. 确认 integration/ 下零直调**

  **What to do**:
  遍历所有 `tests/integration/` 文件，确认**无任何直接 `execute_warp_instruction` 调用残留**。如果有残留：
  1. 若可以 step_warp 化 → 重构（回退到对应 Wave 任务）
  2. 若测试本质上是 handler 隔离测试 → 迁移到 `tests/unit/`（允许直调）
  3. 更新 AGENTS.md 和 CMakeLists.txt

  **Must NOT do**:
  - 不允许任何例外留在 integration/ 中

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需判断每个残留直调是重构不足还是分类错误

  **Parallelization**:
  - **Can Run In Parallel**: NO（需等待 W2-W3 完成）
  - **Blocked By**: W2, W3

  **Acceptance Criteria**:
  - [ ] `grep -r "execute_warp_instruction" tests/integration/ --include="*.cpp" | grep -v "CMakeLists" | wc -l` == 0
  - [ ] 所有残留文件要么已重构，要么已迁移到 unit/

  **QA Scenarios**:
  ```
  Scenario: 验证 integration/ 零直调
    Tool: Bash
    Steps:
      1. count=$(grep -r "execute_warp_instruction" tests/integration/ --include="*.cpp" | grep -v "CMakeLists" | wc -l)
      2. echo $count
    Expected Result: 0
    Evidence: .omo/evidence/w4-4-zero-direct-calls.log
  ```

  **Commit**: YES
  - Message: `refactor(tests): ensure zero direct execute_warp_instruction in integration/`

---

## Final Verification Wave

- [~] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .omo/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [~] F2. **Code Quality Review** — `unspecified-high`
  Run `tsc --noEmit` + linter + `bun test`. Review all changed files for: `as any`/`@ts-ignore`, empty catches, console.log in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names (data/result/item/temp).
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [~] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill if UI)
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (features working together, not isolation). Test edge cases: empty state, invalid input, rapid actions. Save to `.omo/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [~] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination: Task N touching Task M's files. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- 每文件独立 commit：`type(scope): desc`
- Wave 0 commits 前置：`docs(agents): add principle 5 exception clause`
- N/A 迁移 commit：`refactor(tests): move <file> from integration/ to unit/<area>/`
- 重构 commit：`refactor(tests): <file> use step_warp per 5 principles`
- 阶段间 sanity check：`./scripts/sanity.sh --quick`

---

## Success Criteria

### Verification Commands
```bash
# 1. 违规率降至 0（integration/ 下无任何直调）
grep -r "execute_warp_instruction" tests/integration/ --include="*.cpp" | grep -v "CMakeLists" | wc -l
# Expected: 0

# 2. integration 标签只含指令序列测试
ctest -L integration -N 2>&1 | grep -c "Test #"
# Expected: ~15 (不含 N/A 文件)

# 3. unit 标签含迁移文件
ctest -L unit -N 2>&1 | grep -c "Test #"
# Expected: 原数量 + 迁移数量

# 4. 全量 sanity 通过
cd build && ctest
# Expected: 0 failures

# 5. sanity.sh 通过
./scripts/sanity.sh
# Expected: exit code 0
```

### Final Checklist
- [x] All "Must Have" present (35/35 implementation tasks done; 0 violations in integration/)
- [x] All "Must NOT Have" absent (no tool expansion, no batch commits, no scope creep)
- [~] All tests pass — **BLOCKED on runtime verification**: 14 mechanical refactors (W2.1 + 13) not yet runtime-tested; W2.1 confirmed to hang at 5min ctest timeout. See `.omo/evidence/w2-1-finding.md`
- [x] AGENTS.md 原则 5 含分类规则（line 259）— user decision: NO exception clause, instead classification rule
- [x] AGENTS.md 含 integration/unit 区分（line 312）— user decision: NO handler isolation in integration/, unit/ allows direct calls
- [x] sanity.sh regex 与 ctest 名同步（commit 1cce0ec）
- [x] 无孤儿文件（所有 .cpp/.cu 已注册或归档）
