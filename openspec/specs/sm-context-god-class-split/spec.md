# sm-context-god-class-split Specification

## Purpose
TBD - created by archiving change god-class-refactor-sm-context. Update Purpose after archive.
## Requirements
### Requirement: 拆分 ≤ 4 组件

The system MUST split sm_context.cpp into ≤ 4 components.

#### Scenario: 组件数 ≤ 4

- **WHEN** 提取后运行 `ls src/ptxsim/core/sm_context_*.{h,cpp} | grep -v '^src/ptxsim/core/sm_context\.[ch]pp$'`
- **THEN** 命中 ≤ 8（4 组件 × .h+.cpp）

#### Scenario: 主文件行数 < 250

- **WHEN** 拆分后运行 `wc -l src/ptxsim/core/sm_context.cpp`
- **THEN** 命中 < 250

### Requirement: exe_once 主循环签名不变

The system MUST NOT change `exe_once()` main loop signature.

#### Scenario: 编译期验证

- **WHEN** 拆分后 `cmake --build build`
- **THEN** 无 `exe_once()` signature mismatch 错误

### Requirement: 3 Phase commit

The system MUST split work into 3 independent commits per Checklist B.

#### Scenario: 独立可 revert

- **GIVEN** 3 个 Phase commit
- **WHEN** 任意 Phase 失败
- **THEN** 可 `git revert HEAD` 独立回滚
- **AND** 不影响其他 Phase

