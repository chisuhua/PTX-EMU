# PC API — Dead Code Cleanup Spec

## ADDED Requirements

### Requirement: Zero-Production-Refs

`ThreadContext::force_set_pc`、`WarpContext::get_pc`、`WarpContext::set_pc`、
`WarpContext::pc` 在**生产代码** (`src/` + `include/`，**不包含** `tests/`)
中零引用。

#### Scenario: verify-zero-refs-production

- **WHEN** 运行
  ```bash
  grep -rnE "force_set_pc|WarpContext::(get_pc|set_pc)|warp\.(get_pc|set_pc)\s*\(" \
    --include="*.cpp" --include="*.h" src/ include/ \
    | grep -v "build/" | grep -v "antlr4_generated_src/"
  ```
- **THEN** 输出为空（仅可能命中 `warp_context.h` 注释占位 "Removed 2026-07-XX"）

### Requirement: Clean-Build-No-Warnings

删除 deprecated API 后，**clean build** 不得产生 `[[deprecated]]` 调用的
warning 或编译错误。

#### Scenario: verify-clean-build

- **WHEN** 运行
  ```bash
  rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j$(nproc) 2>&1 | tee /tmp/build.log
  grep -iE "deprecated|warning" /tmp/build.log \
    | grep -v "bench/" | grep -v "antlr4_generated_src/" \
    | grep -v "Removed 2026-07"
  ```
- **THEN** 输出为空

### Requirement: No-Regression

所有现有测试（unit + integration + e2e + PTX 语法）无新增 FAIL。

#### Scenario: verify-no-regression

- **WHEN** 运行
  ```bash
  cd build && ctest --output-on-failure
  ./scripts/sanity.sh --quick
  ./tests/ptx/test_all_ptx.sh
  ```
- **THEN** 全部 PASS，且与 baseline (`/tmp/sanity-baseline.txt`) 对比无新增 FAIL

### Requirement: Doc-Sync-Resolved

ADR-0003、ADR-0008、`src/ptxsim/core/AGENTS.md`、
`src/ptxsim/instructions/AGENTS.md` 不再推荐 `force_set_pc`。

#### Scenario: verify-doc-sync

- **WHEN** 运行
  ```bash
  grep -rn "force_set_pc" docs/adr/0003-commit-pc-pattern.md \
    docs/adr/0008-barrier-semantics.md \
    src/ptxsim/core/AGENTS.md \
    src/ptxsim/instructions/AGENTS.md
  ```
- **THEN** 仅命中 "Removed 2026-07-XX" 或 "历史实现" 字样，**无** "use
  `force_set_pc`" 推荐

## REMOVED Requirements

### `ThreadContext::force_set_pc(int)`

- **移除原因**：
  1. 生产代码零调用（仅 `tests/unit/pc/test_pc_management.cpp` 4 处引用，
     全部可重写或删除）
  2. `set_pc()` 同时写 `pc` + `next_pc` 已覆盖初始化 / 同步 / 重置场景
  3. barrier 完成路径已改用 `advance_thread_pc()`（commit `5f7c8a2`）
  4. 保留 `force_set_pc` 会持续产生编译警告（`[[deprecated]]` 调用）

### `WarpContext::get_pc() / set_pc() / int pc`

- **移除原因**：
  1. 生产代码零调用
  2. SIMT v2.0（commit `5f7c8a2`）后每线程 PC（`warp_state.threads[i].pc`）
     是权威源
  3. warp 级 PC 字段自引入以来就是 dead field，仅在 `reset()` 中被重置，
     从未被业务代码读取
  4. `advance_thread_pc(lane, pc)` 是 WarpContext 唯一对外的 PC 写入入口

## Negative Requirements

- **不得**修改 `ThreadContext::set_pc()` / `commit_pc()` / `get_pc()` /
  `get_next_pc()` 任何实现细节
- **不得**修改 `WarpContext::advance_thread_pc()` 实现
- **不得**修改 barrier / branch / dispatch 等 PC 设置逻辑
- **不得**删除 `WarpContext::set_thread_pc()`（29 处测试调用点，超出范围）
- **不得**删除 `WarpState::threads[i].pc`（权威源）
- **不得**实施 `set_thread_pc → advance_thread_pc` 的迁移（单独 change）
- **不得**删除 `Wbar` 相关任何内容（C2 已完成）
- **不得**在单个 commit 中混合代码删除、测试重写、文档同步
