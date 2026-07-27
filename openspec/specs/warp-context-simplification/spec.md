# warp-context-simplification Specification

## Purpose
TBD - created by archiving change refactor-warp-context. Update Purpose after archive.
## Requirements
### Requirement: 主文件行数 ≤ 300

The system MUST reduce `warp_context.cpp` to less than 300 lines by extracting 3 helper modules.

#### Scenario: 拆分后主文件行数

- **WHEN** 提取完成后运行 `wc -l src/ptxsim/core/warp_context.cpp`
- **THEN** 命中 < 300

#### Scenario: 新增 3 个子模块

- **WHEN** 提取完成后运行 `ls src/ptxsim/core/warp_context_*.{h,cpp}`
- **THEN** 命中 ≥ 6（3 模块 × .h+.cpp）

### Requirement: 行为不变性

The system MUST maintain identical execution/sync/convergence semantics.

#### Scenario: execute_warp_instruction 行为一致

- **WHEN** 运行 ctest 含 execute_warp_instruction 路径
- **THEN** 零回归

#### Scenario: 4 处 sync_to_warp_state 完整保留

- **GIVEN** lessons-learned §1 跨模块状态翻译（4 站点：warp_context.cpp:337/:345/:370/:375）
- **WHEN** 提取后
- **THEN** 4 处 `sync_to_warp_state()` 调用在主文件或新模块中仍存在
- **AND** 调用顺序不变

