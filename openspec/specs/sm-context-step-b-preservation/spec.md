# sm-context-step-b-preservation Specification

## Purpose
TBD - created by archiving change god-class-refactor-sm-context. Update Purpose after archive.
## Requirements
### Requirement: step_b_set_blocked_cycles 4 分支测试锁定

The system MUST preserve the 4-branch test coverage for `step_b_set_blocked_cycles`.

#### Scenario: 单元测试全绿

- **GIVEN** `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` 4 分支测试
- **WHEN** step_b 移动到新文件
- **THEN** 4 分支测试全绿
- **AND** 测试代码无需修改

#### Scenario: byte-identical fallback 行为

- **GIVEN** no-op fallback 必须字节级一致
- **WHEN** step_b 被任意调用
- **THEN** 输出与移动前字节级一致

