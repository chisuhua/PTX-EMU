# warp-context-api-freeze Specification

## Purpose
TBD - created by archiving change refactor-warp-context. Update Purpose after archive.
## Requirements
### Requirement: 4 个 WarpContext public 方法签名冻结

The system MUST preserve signatures of 4 public methods without modification.

#### Scenario: update_active_mask 签名不变

- **GIVEN** 消费方 `sm_context.cpp:379` 调用 `w->update_active_mask()`
- **WHEN** 提取 active mask helper 后
- **THEN** 签名 `void update_active_mask()` 保持不变
- **AND** 编译期通过（无参数变更）

#### Scenario: check_reconvergence 签名不变

- **GIVEN** 消费方 `sm_context.cpp:468` 与 `:590` 调用
- **WHEN** 提取 SIMT 编排后
- **THEN** 签名 `bool check_reconvergence()` 保持不变

#### Scenario: get_simt_stack 签名不变

- **GIVEN** 消费方 `sm_context.cpp:461` 与 `:583` 调用
- **WHEN** 提取后
- **THEN** 签名 `SimtStack& get_simt_stack()` 保持不变

#### Scenario: get_lanes_by_pc 签名不变

- **GIVEN** 消费方 `sm_context.cpp:489`（隐式）调用
- **WHEN** 提取后
- **THEN** 签名不变

### Requirement: 编译期 API 冻结证据

The system MUST verify sm_context.cpp 零 diff after extraction.

#### Scenario: sm_context.cpp 编译通过且零 diff

- **WHEN** 提取完成后
- **THEN** `git diff src/ptxsim/core/sm_context.cpp` 无任何变更
- **AND** `cmake --build build` 成功

