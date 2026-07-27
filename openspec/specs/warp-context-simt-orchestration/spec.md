# warp-context-simt-orchestration Specification

## Purpose
TBD - created by archiving change refactor-warp-context. Update Purpose after archive.
## Requirements
### Requirement: SIMT 编排逻辑提取

The system MUST extract the SIMT orchestration logic (divergence/reconvergence handling) to `warp_context_simt.{h,cpp}`.

#### Scenario: 编排逻辑在新模块

- **WHEN** 提取后运行 `grep -c 'check_reconvergence\|simt_stack.push\|simt_stack.pop' src/ptxsim/core/warp_context.cpp`
- **THEN** 命中 ≤ 0（编排调用已抽离）
- **AND** 在新模块中保留

#### Scenario: simt_stack 数据结构未变

- **GIVEN** simt_stack.cpp 数据结构已抽离
- **WHEN** 提取编排逻辑后
- **THEN** simt_stack.cpp 零 diff
- **AND** simt_stack.h 零 diff

### Requirement: set_active_mask overwrite 语义

The system MUST preserve set_active_mask overwrite semantics (not OR merge).

#### Scenario: ret handler 行为不变

- **GIVEN** ret handler 依赖 overwrite 语义
- **WHEN** set_active_mask 被调用
- **THEN** 当前活动 mask 完全替换
- **AND** ret handler 测试零回归

### Requirement: 4 处 sync_to_warp_state 行级保留

The system MUST preserve 4 `thread->sync_to_warp_state()` calls during migration.

#### Scenario: §1 行级 diff 核对

- **GIVEN** lessons-learned §1 跨模块状态翻译（4 站点）
- **WHEN** 提取后
- **THEN** warp_context.cpp:337/:345/:370/:375（或迁移后新位置）保留 4 处调用
- **AND** 调用顺序与原文件一致

